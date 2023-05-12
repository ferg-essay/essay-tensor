pub trait Dataset<T> {
    /*
    fn apply<F>(&self, fun: F) -> dyn Dataset
    where F: Fn(dyn Dataset) -> dyn Dataset;
    */
    fn iter(&self) -> dyn Iterator<Item=T>;

    fn get_single_element(&self) -> T;

    //fn interleave<S>(&self, map: Fn(T) -> Dataset<S>) -> Dataset<S>;

}

type BoxDataset<T> = Box<dyn Dataset<T>>;

fn list_files<T>(file_pattern: &str) -> Box<dyn Dataset<T>> {
    todo!()
}

fn random(seed: Option<usize>) -> Box<dyn Dataset<f32>> {
    todo!()
}

fn range(args: &[usize]) -> Box<dyn Dataset<f32>> {
    todo!()
}

fn sample_from_datasets(datasets: Vec<BoxDataset<f32>>) -> BoxDataset<f32> {
    todo!()
}