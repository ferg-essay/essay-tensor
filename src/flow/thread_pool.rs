use core::{fmt, panic};
use std::{
    thread::{self, JoinHandle}, 
    sync::{mpsc::{self}, Arc}, any::Any, marker::PhantomData,
};

use concurrent_queue::{ConcurrentQueue, PopError, PushError};
use log::info;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MainId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChildId(usize);

pub type Result<T> = std::result::Result<T, ThreadPoolErr>;

pub enum Out<T> {
    None,
    Some(T),
    Pending,
}

pub trait Main<T, MP: Msg, PM: Msg> {
    fn on_start(&mut self, to_parent: &mut dyn Sender<MP>) -> Result<()>;
    fn on_parent(&mut self, msg: PM, to_parent: &mut dyn Sender<MP>) -> Result<Out<T>>;
}

pub trait Parent<MP: Msg, PM: Msg, PC: Msg, CP: Msg> : Send {
    fn on_start(&mut self);

    fn on_main_start(&mut self, id: MainId);
    fn on_main(
        &mut self, 
        id: MainId, 
        msg: MP, 
        to_main: &mut dyn Sender<PM>, 
        to_child: &mut dyn Sender<PC>
    ) -> Result<()>;
    fn on_main_end(&mut self, id: MainId);

    fn on_child(
        &mut self, 
        msg: CP, 
        to_main: &mut dyn Sender<PM>, 
        to_child: &mut dyn Sender<PC>
    ) -> Result<()>;
}

pub trait Child<PC: Msg, CP: Msg> : Send {
    fn on_start(&mut self);
    fn on_parent(&mut self, msg: PC, to_parent: &mut dyn Sender<CP>) -> Result<()>;
}

type ChildBuilder<PC, CP> = dyn FnMut() -> Box<dyn Child<PC, CP> + Send>;

#[derive(Debug)]
enum MainMessage<PC> {
    OnParent(PC),
    Exit,
    Panic,
}

#[derive(Debug)]
enum ParentMessage<MP, CP> {
    OnMain(MainId, MP),
    OnChild(ChildId, CP),
    Exit,
    Panic,
}

#[derive(Debug)]
enum ChildMessage<PC> {
    OnParent(PC),
}

#[derive(Debug)]
pub enum ThreadPoolErr {
    Err(Box<dyn Any + Send>),
    SendError,
    RecvErr(mpsc::RecvError),
    ChildPanic,
}

pub trait Sender<M: Msg> {
    fn send(&mut self, msg: M) -> Result<()>;
}

pub struct ChannelSender<M: Msg>(mpsc::Sender<M>);

pub trait Msg : fmt::Debug + Send + 'static {}

//
// ThreadPool -- Caller (Main) Implementation
//

pub struct ThreadPool<T, MP: Msg, PM: Msg, PC: Msg, CP: Msg> {
    parent: Option<JoinHandle<()>>,

    receiver: mpsc::Receiver<MainMessage<PM>>,
    to_parent: mpsc::Sender<ParentMessage<MP, CP>>,

    marker: PhantomData<(T, PC)>,
}

impl<T, MP: Msg, PM: Msg, PC: Msg, CP: Msg> ThreadPool<T, MP, PM, PC, CP> {
    pub fn builder() -> ThreadPoolBuilder<T, MP, PM, PC, CP> {
        ThreadPoolBuilder::new()
    }

    pub fn start(&self, main: impl Main<T, MP, PM>) -> Result<Option<T>> {
        let mut main = main;

        let id = MainId(0);
        let mut to_parent = MainToParent(id, &self.to_parent);

        main.on_start(&mut to_parent)?;
        // self.parent_sender.send(ParentMessage::OnMain(msg));
        
        loop {
            match self.receiver.recv() {
                Ok(MainMessage::OnParent(msg)) => {
                    match main.on_parent(msg, &mut to_parent) {
                        Ok(Out::None) => return Ok(None),
                        Ok(Out::Some(value)) => return Ok(Some(value)),
                        Ok(Out::Pending) => continue,
                        Err(err) => return Err(err),
                    }
                }
                Ok(MainMessage::Exit) => {
                    panic!("unexpected exit");
                }
                Ok(MainMessage::Panic) => {
                    panic!("parent panic received by thread pool");
                }
                Err(err) => {
                    println!("executor receive error {:?}", err);
                    return Err(ThreadPoolErr::RecvErr(err));
                }
            }
        }
    }

    pub fn close(&mut self) -> Result<()> {
        match self.parent.take() {
            Some(handle) => {
                match self.to_parent.send(ParentMessage::Exit) {
                    Ok(_) => {},
                    Err(err) => {
                        info!("error sending exit {:#?}", err);
                    },
                };

                // TODO: timed?
                match handle.join() {
                    Ok(_) => Ok(()),
                    Err(err) => Err(ThreadPoolErr::Err(err)),
                }
            },
            None => Ok(()),
        }
    }
}

impl<T, MP: Msg, PM: Msg, CP: Msg, PC: Msg> Drop for ThreadPool<T, MP, PM, CP, PC> {
    fn drop(&mut self) {
        match self.close() {
            Ok(_) => {},
            Err(err) => { info!("error while closing {:#?}", err) }
        };
    }
}

struct MainToParent<'a, MP, CP>(MainId, &'a mpsc::Sender<ParentMessage<MP, CP>>);

impl<MP: Msg, CP: Msg> Sender<MP> for MainToParent<'_, MP, CP> {
    fn send(&mut self, msg: MP) -> Result<()> {
        match self.1.send(ParentMessage::OnMain(self.0, msg)) {
            Ok(_) => { Ok(()) },
            Err(_) => { panic!("failed send"); }
        }
    }
}

//
// Parent thread
//

pub struct ParentThread<MP: Msg, PM: Msg, PC: Msg, CP: Msg> {
    task: Box<dyn Parent<MP, PM, PC, CP>>,

    receiver: mpsc::Receiver<ParentMessage<MP, CP>>,
    to_main: mpsc::Sender<MainMessage<PM>>,

    child_pool: Arc<ChildQueues<PC>>,

    handles: Vec<JoinHandle<()>>,
}

impl<MP: Msg, PM: Msg, CP: Msg, PC: Msg> ParentThread<MP, PM, CP, PC> {
    pub fn run(&mut self) -> Result<()> {
        let mut guard = ParentGuard::new(&self.to_main);

        let mut to_main = ToMain(&self.to_main);
        let mut to_child = ToChild::new(&self.child_pool);

        loop {
            match self.receiver.recv() {
                Ok(ParentMessage::OnMain(id, msg)) => {
                    self.task.on_main(id, msg, &mut to_main, &mut to_child)?;
                }
                Ok(ParentMessage::OnChild(_id, msg)) => {
                    self.task.on_child(msg, &mut to_main, &mut to_child)?;
                }
                Ok(ParentMessage::Exit) => {
                    to_child.close();
                    self.to_main.send(MainMessage::Exit).unwrap();
                    guard.close();
                    return Ok(());
                }
                Ok(_) => {
                    // self.to_main.send(MainMessage::Panic).unwrap();
                    panic!("invalid executor message");
                }
                Err(err) => {
                    // self.to_main.send(MainMessage::Panic).unwrap();
                    panic!("executor receive error {:?}", err);
                }
            }

            self.unpark();
        }
    }

    fn unpark(&self) {
        for h in &self.handles {
            h.thread().unpark();
        }
    }
}

struct ToMain<'a, PM: Msg>(&'a mpsc::Sender<MainMessage<PM>>);

impl<M: Msg> Sender<M> for ToMain<'_, M> {
    fn send(&mut self, msg: M) -> Result<()> {
        match self.0.send(MainMessage::OnParent(msg)) {
            Ok(_) => { Ok(()) },
            Err(_) => { panic!("failed send"); }
        }
    }
}

struct ChildQueues<PC> {
    child_queue: ConcurrentQueue<ChildMessage<PC>>,
    tasks: Vec<ChildInfo>,
}

impl<PC> ChildQueues<PC> {

}

struct ChildInfo {
    _handle: Option<JoinHandle<()>>,
}

impl ChildInfo {
    pub fn new() -> Self {
        ChildInfo {
            _handle: None,
        }
    }
}

struct ToChild<PC: Msg> {
    child_pool: Arc<ChildQueues<PC>>,
}

impl<PC: Msg> ToChild<PC> {
    fn new(child_pool: &Arc<ChildQueues<PC>>) -> Self {
        Self {
            child_pool: child_pool.clone()
        }
    }

    fn close(&self) {
        self.child_pool.child_queue.close();
    }
}

impl<PC: Msg> Sender<PC> for ToChild<PC> {
    fn send(&mut self, msg: PC) -> Result<()> {
        self.child_pool.child_queue.push(ChildMessage::OnParent(msg))?;

        Ok(())
    }
}

impl<PC: Msg> Drop for ToChild<PC> {
    fn drop(&mut self) {
        self.child_pool.child_queue.close();
    }
}

struct ParentGuard<PM: Msg> {
    to_main: mpsc::Sender<MainMessage<PM>>,
    is_close: bool,
}

impl<PM: Msg> ParentGuard<PM> {
    fn new(to_main: &mpsc::Sender<MainMessage<PM>>) -> Self {
        Self {
            to_main: to_main.clone(),
            is_close: false,
        }
    }

    fn close(&mut self) {
        self.is_close = true;
    }
}

impl<PM: Msg> Drop for ParentGuard<PM> {
    fn drop(&mut self) {
        if ! self.is_close {
            self.to_main.send(MainMessage::Panic).unwrap();
        }
    }
}

//
// ChildThread
// 

struct ChildThread<MP, PC, CP> {
    id: ChildId,

    task: Box<dyn Child<PC, CP>>,

    child_queues: Arc<ChildQueues<PC>>,

    to_parent: Option<mpsc::Sender<ParentMessage<MP, CP>>>,
}

impl<MP: Msg, PC: Msg, CP: Msg> ChildThread<MP, PC, CP> {
    pub fn new(
        id: ChildId,
        task: Box<dyn Child<PC, CP> + Send + 'static>,
        child_queues: Arc<ChildQueues<PC>>, 
        to_parent: mpsc::Sender<ParentMessage<MP, CP>>,
    ) -> Self {
        Self {
            id,
            task,
            child_queues,
            to_parent: Some(to_parent),
        }
    }

    pub fn run(&mut self) {
        let sender = self.to_parent.take().unwrap();
        let mut to_parent = ChildToParent(self.id, sender.clone());
        let _guard = ChildGuard::new(sender);

        let queue = &self.child_queues.child_queue;

        loop {
            match queue.pop() {
                Ok(ChildMessage::OnParent(msg)) => {
                    self.task.on_parent(msg, &mut to_parent).unwrap();
                }
                Err(PopError::Empty) => {
                    thread::park();
                    continue;
                }
                Err(_err) => { return; } // Queue Closed panic!("unknown queue error {:?}", err)
            }
        }
    }
}

struct ChildToParent<MP, CP>(ChildId, mpsc::Sender<ParentMessage<MP, CP>>);

impl<MP: Msg, CP: Msg> Sender<CP> for ChildToParent<MP, CP> {
    fn send(&mut self, msg: CP) -> Result<()> {
        match self.1.send(ParentMessage::OnChild(self.0, msg)) {
            Ok(_) => { Ok(()) },
            Err(_) => { panic!("failed send"); }
        }
    }
}

struct ChildGuard<MP: Msg, CP: Msg> {
    to_parent: mpsc::Sender<ParentMessage<MP, CP>>,
    is_close: bool,
}

impl<MP: Msg, CP: Msg> ChildGuard<MP, CP> {
    fn new(to_parent: mpsc::Sender<ParentMessage<MP, CP>>) -> Self {
        Self {
            to_parent,
            is_close: false,
        }
    }

    fn _close(&mut self) {
        self.is_close = true;
    }
}

impl<MP: Msg, CP: Msg> Drop for ChildGuard<MP, CP> {
    fn drop(&mut self) {
        if ! self.is_close {
            match self.to_parent.send(ParentMessage::Panic) {
                Ok(_) => {},
                Err(_) => {},
            }
        }
    }
}

impl<M: Msg> Sender<M> for ChannelSender<M> {
    fn send(&mut self, msg: M) -> Result<()> {
        match self.0.send(msg) {
            Ok(_) => { Ok(()) },
            Err(_) => { panic!("failed send"); }
        }
    }
}
impl<T> From<PushError<T>> for ThreadPoolErr {
    fn from(_value: PushError<T>) -> Self {
        todo!()
    }
}

//
// ThreadPool builder
//

pub struct ThreadPoolBuilder<T, MP: Msg, PM: Msg, PC: Msg, CP: Msg> {
    parent_task: Option<Box<dyn Parent<MP, PM, PC, CP>>>,
    child_task_builder: Option<Box<ChildBuilder<PC, CP>>>,
    n_threads: Option<usize>,

    marker: PhantomData<(T, PM)>,
}

impl<T, MP: Msg, PM: Msg, PC: Msg, CP: Msg> ThreadPoolBuilder<T, MP, PM, PC, CP> {
    pub fn new() -> Self {
        Self {
            parent_task: None,
            child_task_builder: None,
            n_threads: None,
            marker: PhantomData,
        }
    }

    pub fn parent<F>(mut self, parent: F) -> Self
    where
        F: Parent<MP, PM, PC, CP> + 'static
    {
        self.parent_task = Some(Box::new(parent));

        self
    }

    pub fn child<F>(mut self, child: F) -> Self
    where
        F: Fn() -> Box<dyn Child<PC, CP> + Send + 'static> + 'static,
    {
        self.child_task_builder = Some(Box::new(child));

        self
    }

    pub fn n_threads(mut self, n_threads: usize) -> Self {
        assert!(n_threads > 0);

        self.n_threads = Some(n_threads);

        self
    }

    pub fn build(self) -> ThreadPool<T, MP, PM, PC, CP> {
        assert!(! self.parent_task.is_none());
        assert!(! self.child_task_builder.is_none());

        let (to_parent, parent_receiver) = mpsc::channel();
        let (to_main, main_receiver) = mpsc::channel();

        let n_threads = match self.n_threads {
            Some(n_threads) => n_threads,
            None => usize::from(thread::available_parallelism().unwrap()),
        };

        let mut child_pool = ChildQueues {
            child_queue: ConcurrentQueue::unbounded(),
            tasks: Vec::new(),
        };

        for _ in 0..n_threads {
            child_pool.tasks.push(ChildInfo::new());
        }

        let child_pool = Arc::new(child_pool);

        let mut child_handles = Vec::<JoinHandle<()>>::new();

        let mut child_builder = self.child_task_builder.unwrap();

        for i in 0..n_threads {
            let mut task_thread = ChildThread::new(
                ChildId(i),
                child_builder(),
                Arc::clone(&child_pool), 
                to_parent.clone(),
            );

            let handle = thread::spawn(move || {
                task_thread.run();
            });

            child_handles.push(handle);
        }

        let mut parent = ParentThread {
            task: self.parent_task.unwrap(),

            receiver: parent_receiver,
            to_main,

            child_pool,

            handles: child_handles,
        };

        let handle = thread::spawn(move || {
            parent.run().unwrap();
        });

        ThreadPool {
            parent: Some(handle),

            to_parent,
            receiver: main_receiver,

            marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration, sync::{Arc, Mutex}};

    use crate::flow::thread_pool::Sender;

    use super::{Msg, Main, Parent, Child, MainId, Out, Result};

    use super::ThreadPoolBuilder;

    #[test]
    fn two_tasks_two_child_threads() {
        let values = Arc::new(Mutex::new(Vec::<String>::new()));

        let ptr2 = values.clone();
    
        let ptr = values.clone();
        let ptr_child = values.clone();
        let count = 2;
        let mut index = count;
        let mut pool = ThreadPoolBuilder::<String, MP, PM, PC, CP>::new().parent(
            TestParent::new(move |_, _, to_child| {
                ptr.lock().unwrap().push(format!("[MP"));

                for i in 0..index {
                    to_child.send(PC(i))?;
                }

                thread::sleep(Duration::from_millis(150));

                ptr.lock().unwrap().push(format!("MP]"));
                Ok(())
            }, move |_, to_main, _| {
                ptr_child.lock().unwrap().push(format!("[CP"));
                thread::sleep(Duration::from_millis(100));
                ptr_child.lock().unwrap().push(format!("CP]"));

                index -= 1;

                if index == 0 {
                    to_main.send(PM(0))?;
                }

                Ok(())
        })).child(move || {
            let ptr3 = ptr2.clone();

            Box::new(TestChild::new(move |msg: PC, to_parent: &mut dyn Sender<CP>| { 
                ptr3.lock().unwrap().push(format!("[C"));
                thread::sleep(Duration::from_millis(50));
                ptr3.lock().unwrap().push(format!("C]"));
                to_parent.send(CP(msg.0))
            }))
        }).n_threads(2)
        .build();

        let ptr2 = values.clone();
        let ptr3 = values.clone();
        let value = pool.start(TestMain::new(move |to_parent| {
            ptr2.lock().unwrap().push(format!("[Main"));
            to_parent.send(MP(2))
        }, move |msg, _| {
            ptr3.lock().unwrap().push(format!("Main]"));

            Ok(Out::Some(format!("Main[{:?}]", msg)))
        })).unwrap();

        assert_eq!(value, Some("Main[PM(0)]".to_string()));
    
        let list: Vec<String> = values.lock().unwrap().drain(..).collect();
        assert_eq!(list.join(", "), "[Main, [MP, MP], [C, [C, C], C], [CP, CP], [CP, CP], Main]");

        pool.close().unwrap();
    }

    #[test]
    fn two_tasks_one_child_thread() {
        let values = Arc::new(Mutex::new(Vec::<String>::new()));

        let ptr2 = values.clone();
    
        let ptr = values.clone();
        let ptr_child = values.clone();
        let count = 2;
        let mut index = count;
        let mut pool = ThreadPoolBuilder::<String, MP, PM, PC, CP>::new().parent(
            TestParent::new(move |_, _, to_child| {
                ptr.lock().unwrap().push(format!("[MP"));

                for i in 0..index {
                    to_child.send(PC(i))?;
                }

                thread::sleep(Duration::from_millis(150));

                ptr.lock().unwrap().push(format!("MP]"));
                Ok(())
            }, move |_, to_main, _| {
                ptr_child.lock().unwrap().push(format!("[CP"));
                thread::sleep(Duration::from_millis(100));
                ptr_child.lock().unwrap().push(format!("CP]"));

                index -= 1;

                if index == 0 {
                    to_main.send(PM(0))
                } else {
                    Ok(())
                }
        })).child(move || {
            let ptr3 = ptr2.clone();

            Box::new(TestChild::new(move |msg: PC, to_parent: &mut dyn Sender<CP>| { 
                ptr3.lock().unwrap().push(format!("[C"));
                thread::sleep(Duration::from_millis(50));
                ptr3.lock().unwrap().push(format!("C]"));
                to_parent.send(CP(msg.0))
            }))
        }).n_threads(1)
        .build();

        let ptr2 = values.clone();
        let ptr3 = values.clone();
        let value = pool.start(TestMain::new(move |to_parent| {
            ptr2.lock().unwrap().push(format!("[Main"));
            to_parent.send(MP(2))
        }, move |msg, _| {
            ptr3.lock().unwrap().push(format!("Main]"));

            Ok(Out::Some(format!("Main[{:?}]", msg)))
        })).unwrap();

        assert_eq!(value, Some("Main[PM(0)]".to_string()));
    
        let list: Vec<String> = values.lock().unwrap().drain(..).collect();
        assert_eq!(list.join(", "), "[Main, [MP, MP], [C, [C, C], C], [CP, CP], [CP, CP], Main]");

        pool.close().unwrap();
    }

    #[test]
    fn ping_main_parent_one_thread() {
        let values = Arc::new(Mutex::new(Vec::<String>::new()));

        let ptr = values.clone();
        let ptr2 = values.clone();
        let ptr_child = values.clone();

        let mut pool = ThreadPoolBuilder::<String, MP, PM, PC, CP>::new().parent(
            TestParent::new(move |msg, to_main, _| {
                ptr.lock().unwrap().push(format!("{:?}", msg));
                to_main.send(PM(msg.0))
            }, move |_, _, _| {
                ptr2.lock().unwrap().push(format!("CP"));
                panic!("no child");
        })).child(move || {
            let ptr = ptr_child.clone();

            Box::new(TestChild::new(move |_, _| { 
                ptr.lock().unwrap().push(format!("C"));
                panic!("no child");
            }))
        }).n_threads(1)
        .build();

        let ptr2 = values.clone();
        let ptr3 = values.clone();
        let value = pool.start(TestMain::new(move |to_parent| {
            ptr2.lock().unwrap().push(format!("Main"));

            to_parent.send(MP(3))
        }, move |msg, to_parent| {
            ptr3.lock().unwrap().push(format!("{:?}", msg));

            if msg.0 > 0 {
                to_parent.send(MP(msg.0 - 1))?;
                Ok(Out::Pending)
            } else {
                Ok(Out::Some(format!("{:?}", msg)))
            }
        })).unwrap();

        assert_eq!(value, Some("PM(0)".to_string()));
    
        let list: Vec<String> = values.lock().unwrap().drain(..).collect();
        assert_eq!(list.join(", "), "Main, MP(3), PM(3), MP(2), PM(2), MP(1), PM(1), MP(0), PM(0)");

        pool.close().unwrap();
    }

    #[test]
    fn ping_parent_child_one_thread() {
        let values = Arc::new(Mutex::new(Vec::<String>::new()));

        let ptr = values.clone();
        let ptr2 = values.clone();
        let ptr_child = values.clone();

        let mut pool = ThreadPoolBuilder::<String, MP, PM, PC, CP>::new().parent(
            TestParent::new(move |msg, _, to_child| {
                ptr.lock().unwrap().push(format!("{:?}", msg));
                to_child.send(PC(msg.0))?;
                Ok(())
            }, move |msg, to_main, to_child| {
                ptr2.lock().unwrap().push(format!("{:?}", msg));

                if msg.0 > 0 {
                    to_child.send(PC(msg.0 - 1))
                } else {
                    to_main.send(PM(msg.0))
                }
        })).child(move || {
            let ptr = ptr_child.clone();

            Box::new(TestChild::new(move |msg, to_parent| { 
                ptr.lock().unwrap().push(format!("{:?}", msg));
                to_parent.send(CP(msg.0))
            }))
        }).n_threads(1)
        .build();

        let ptr2 = values.clone();
        let ptr3 = values.clone();
        let value = pool.start(TestMain::new(move |to_parent| {
            ptr2.lock().unwrap().push(format!("Main"));
            to_parent.send(MP(3))
        }, move |msg, _| {
            ptr3.lock().unwrap().push(format!("{:?}", msg));

            Ok(Out::Some(format!("{:?}", msg)))
        })).unwrap();

        assert_eq!(value, Some("PM(0)".to_string()));
    
        let list: Vec<String> = values.lock().unwrap().drain(..).collect();
        assert_eq!(list.join(", "), "Main, MP(3), PC(3), CP(3), PC(2), CP(2), PC(1), CP(1), PC(0), CP(0), PM(0)");

        pool.close().unwrap();
    }


    #[test]
    #[should_panic]
    fn panic_in_parent() {
        let values = Arc::new(Mutex::new(Vec::<String>::new()));

        let ptr = values.clone();
        let ptr2 = values.clone();
        let ptr3 = values.clone();

        let mut pool = ThreadPoolBuilder::new().parent(TestParent::new(
            move |_mp, _, _| {
                ptr.lock().unwrap().push(format!("[P"));

                panic!("test parent panic");
            }, move |_, _, _| {
                ptr3.lock().unwrap().push(format!("OnChild"));

                Ok(())
            })).child(move || {
            let ptr3 = ptr2.clone();

            Box::new(TestChild::new(move |_msg, _s| { 
                ptr3.lock().unwrap().push(format!("[C]"));
                Ok(())
            }))
        }).build();

        let ptr2 = values.clone();
        pool.start(TestMain::new(move |to_parent| {
            to_parent.send(MP(0))
        }, move |_msg, _| {
            ptr2.lock().unwrap().push(format!("[OnParent]"));
            Ok(Out::None)
        })).unwrap();

        let list: Vec<String> = values.lock().unwrap().drain(..).collect();
        assert_eq!(list.join(", "), "[P, [C, C], [C, C], P]");

        pool.close().unwrap();
    }

    #[test]
    #[should_panic]
    fn panic_in_child() {
        let values = Arc::new(Mutex::new(Vec::<String>::new()));

        let ptr = values.clone();
        let ptr2 = values.clone();
        let ptr3 = values.clone();

        let mut pool = ThreadPoolBuilder::new().parent(TestParent::new(
            move |_mp, _, to_child| {
                ptr.lock().unwrap().push(format!("[MP]"));

                to_child.send(PC(1))
            }, move |_, _, _| {
                ptr3.lock().unwrap().push(format!("[CP]"));
                Ok(())
            })).child(move || {
            let ptr3 = ptr2.clone();

            Box::new(TestChild::new(move |_msg, _s| { 
                ptr3.lock().unwrap().push(format!("[C"));
                panic!("Panic in child");
            }))
        }).build();

        let ptr2 = values.clone();
        pool.start(TestMain::new(move |to_parent| {
            to_parent.send(MP(0))
        }, move |_msg, _| {
            ptr2.lock().unwrap().push(format!("[PM]"));
            Ok(Out::None)
        })).unwrap();

        let list: Vec<String> = values.lock().unwrap().drain(..).collect();
        assert_eq!(list.join(", "), "[P, [C, C], [C, C], P]");

        pool.close().unwrap();
    }

    struct TestMain {
        on_start: Box<dyn FnMut(&mut dyn Sender<MP>) -> Result<()>>,
        on_parent: Box<dyn FnMut(PM, &mut dyn Sender<MP>) -> Result<Out<String>>>,
    }

    impl TestMain {
        fn new(
            on_start: impl FnMut(&mut dyn Sender<MP>) -> Result<()> + 'static,
            on_parent: impl FnMut(PM, &mut dyn Sender<MP>) -> Result<Out<String>> + 'static
        ) -> Self {
            Self {
                on_start: Box::new(on_start),
                on_parent: Box::new(on_parent),
            }
        }
    }

    impl Main<String, MP, PM> for TestMain {
        fn on_start(&mut self, to_parent: &mut dyn Sender<MP>) -> Result<()> {
            (self.on_start)(to_parent)
        }

        fn on_parent(&mut self, msg: PM, to_parent: &mut dyn Sender<MP>) -> Result<Out<String>> {
            (self.on_parent)(msg, to_parent)
        }
    }
    struct TestParent {
        on_main: Box<dyn FnMut(MP, &mut dyn Sender<PM>, &mut dyn Sender<PC>) -> Result<()> + Send>,
        on_child: Box<dyn FnMut(CP, &mut dyn Sender<PM>, &mut dyn Sender<PC>) -> Result<()> + Send>,
    }

    impl TestParent {
        fn new(
            on_main: impl FnMut(MP, &mut dyn Sender<PM>, &mut dyn Sender<PC>) -> Result<()> + Send + 'static,
            on_child: impl FnMut(CP, &mut dyn Sender<PM>, &mut dyn Sender<PC>) -> Result<()> + Send + 'static,
        ) -> Self {
            Self {
                on_main: Box::new(on_main),
                on_child: Box::new(on_child),
            }
        }
    }

    impl Parent<MP, PM, PC, CP> for TestParent {
        fn on_start(&mut self) {
            todo!()
        }

        fn on_main_start(&mut self, _id: super::MainId) {
            todo!()
        }

        fn on_main(
            &mut self, 
            _id: MainId, 
            msg: MP, to_main: 
            &mut dyn Sender<PM>, 
            to_child: &mut dyn Sender<PC>) -> Result<()> {
            (self.on_main)(msg, to_main, to_child)
        }

        fn on_main_end(&mut self, _id: super::MainId) {
            todo!()
        }

        fn on_child(
            &mut self, 
            msg: CP, 
            to_main: &mut dyn Sender<PM>, 
            to_child: &mut dyn Sender<PC>
        ) -> Result<()> {
            (self.on_child)(msg, to_main, to_child)
        }
    }
    struct TestChild {
        on_parent: Box<dyn FnMut(PC, &mut dyn Sender<CP>) -> Result<()> + Send>,
    }

    impl TestChild {
        fn new(
            on_parent: impl FnMut(PC, &mut dyn Sender<CP>) -> Result<()> + Send + 'static
        ) -> Self {
            Self {
                on_parent: Box::new(on_parent),
            }
        }
    }

    impl Child<PC, CP> for TestChild {
        fn on_start(&mut self) {
            todo!()
        }

        fn on_parent(&mut self, msg: PC, to_parent: &mut dyn Sender<CP>) -> Result<()> {
            (self.on_parent)(msg, to_parent)
        }
    }

    #[derive(Debug)]
    struct MP(usize);
    impl Msg for MP {}

    #[derive(Debug)]
    struct PM(usize);
    impl Msg for PM {}

    #[derive(Debug)]
    struct PC(usize);
    impl Msg for PC {}

    #[derive(Debug)]
    struct CP(usize);
    impl Msg for CP {}
}