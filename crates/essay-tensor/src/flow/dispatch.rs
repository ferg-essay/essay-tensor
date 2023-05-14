use crossbeam::channel::{self, Sender, Receiver};


struct Executor {
    sender: Sender<TaskOp>,
    receiver: Receiver<TaskOp>,
}

enum TaskOp {
    Exit,
}

impl Executor {
    fn new() -> Self {
        let (sender, receiver) = channel::unbounded::<TaskOp>();

        Self {
            sender,
            receiver,
        }
    }
}
#[cfg(test)]
mod test {

    #[test]    
    fn test_async() {
        // let future = async_fun();
        // println!("Future {:?}", future);
    }
    
}