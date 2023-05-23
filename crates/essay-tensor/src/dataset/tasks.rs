use std::collections::VecDeque;

trait Source<T> {
    fn next(&mut self) -> Option<T>;
}

struct Count {
    count: usize,
}

impl Count {
    pub fn new(count: usize) -> Self {
        Self {
            count,
        }
    }
}

impl Source<usize> for Count {
    fn next(&mut self) -> Option<usize> {
        if self.count > 0 {
            let value = self.count;
            self.count -= 1;
            Some(value)
        } else {
            None
        }
    }
}

struct QueueSource<T> {
    queue: VecDeque<T>,
    next: Box<dyn Source<T>>,
}

impl<T> QueueSource<T> {
    pub fn new(next: impl Source<T> + 'static) -> Self {
        Self {
            queue: Default::default(),
            next: Box::new(next),
        }
    }
}

impl<T> QueueSource<T> {
    async fn my_next(&mut self) -> Option<T> {
        let future = async { 
            self.next.next() 
        };

        future.await
    }
}

struct Middle<T> {
    next: Box<dyn Source<T>>,
}

impl<T: 'static> Middle<T> {
    pub fn new(next: impl Source<T> + 'static) -> Self {
        Self {
            next: Box::new(next),
        }
    }
}

impl<T> Source<T> for Middle<T> {
    fn next(&mut self) -> Option<T> {
        self.next.next()
    }
}

#[cfg(test)]
mod test {
    use futures::executor;

    use crate::dataset::tasks::{Count, Middle, Source};

    #[test]
    fn test() {
        println!("Hello");
        let count = Count::new(5);
        let mut middle = Middle::new(count);

        let future = async { 
            while let Some(v) = middle.next() {
                println!("value {}", v); 
            }
        };
        executor::block_on(future);
    }
}