mod nn;
#[cfg(test)]
mod test {
    use ndarray::array;

    #[test]
    fn test() {
        let a = array![
            [1., 0., 2.],
            [1., 0., 2.],
        ];
        let b = a.clone();
        println!("hello {:?}", a);
        println!("h2 {:?}", &a.dot(&b.t()));
    }
}