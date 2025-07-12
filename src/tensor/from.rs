use super::{data::TensorData, Shape, Tensor, Type};


impl<T: Type> From<&Tensor<T>> for Tensor<T> {
    fn from(value: &Tensor<T>) -> Self {
        value.clone()
    }
}

impl<T: Type> From<T> for Tensor<T> {
    fn from(value: T) -> Self {
        Tensor::from_vec(vec![value], Shape::scalar())
    }
}

impl<T: Type> From<Option<T>> for Tensor<T> {
    fn from(value: Option<T>) -> Self {
        match value {
            Some(value) => Tensor::from_vec(vec![value], Shape::scalar()),
            None => {
                let v: [T; 0] = [];
                Tensor::from_vec(Vec::from(v), [0])
            }
        }
    }
}

// vec conversions

impl<T: Type> From<Vec<T>> for Tensor<T> {
    fn from(value: Vec<T>) -> Self {
        let len = value.len();

        Tensor::<T>::from_vec(value, Shape::from(len))
    }
}

impl<T: Type + Clone> From<&Vec<T>> for Tensor<T> {
    fn from(value: &Vec<T>) -> Self {
        let data = value.clone();
        let len = data.len();

        Tensor::from_vec(data, len)
    }
}

impl<T: Type, const N: usize> From<Vec<[T; N]>> for Tensor<T> {
    fn from(value: Vec<[T; N]>) -> Self {
        let len = value.len();

        let slice = value.into_boxed_slice();

        TensorData::<T>::from_boxed_rows(slice, [len, N])
    }
}

impl<T: Type + Clone, const N: usize> From<&Vec<[T; N]>> for Tensor<T> {
    fn from(value: &Vec<[T; N]>) -> Self {
        let len = value.len();

        let data: Vec<T> = value.iter()
            .flat_map(|row| row.iter().map(|v| v.clone()))
            .collect();

        Tensor::from_vec(data, [len, N])
    }
}

impl<T: Type + Clone> From<&Vec<Vec<T>>> for Tensor<T> {
    fn from(value: &Vec<Vec<T>>) -> Self {
        let len_1 = value[0].len();

        let data = value.iter().flat_map(|v1| {
            assert_eq!(len_1, v1.len());

            v1.iter().map(|v2| v2.clone())
        }).collect();

        Tensor::from_vec(data, [value.len(), len_1])
    }
}

impl<T: Type + Clone, const N: usize> From<&Vec<Vec<[T; N]>>> for Tensor<T> {
    fn from(value: &Vec<Vec<[T; N]>>) -> Self {
        let len_1 = value[0].len();

        let data = value.iter().flat_map(|v1| {
            assert_eq!(len_1, v1.len());

            v1.iter().flat_map(|v2| {
                v2.iter().map(|v3| v3.clone())
            })
        }).collect();

        Tensor::from_vec(data, [value.len(), len_1, N])
    }
}

impl<T: Clone + Type> From<&Vec<Vec<Vec<T>>>> for Tensor<T> {
    fn from(value: &Vec<Vec<Vec<T>>>) -> Self {
        let mut data = Vec::new();

        let len_1 = value[0].len();
        let len_2 = value[0][0].len();

        for v1 in value {
            assert_eq!(len_1, v1.len());

            for v2 in v1 {
                assert_eq!(len_2, v2.len());

                for v3 in v2 {
                    data.push(v3.clone());
                }
            }
        }

        Tensor::from_vec(data, [value.len(), len_1, len_2])
    }
}

impl<T: Clone + Type> From<&Vec<Vec<Vec<Vec<T>>>>> for Tensor<T> {
    fn from(value: &Vec<Vec<Vec<Vec<T>>>>) -> Self {
        let mut data = Vec::new();

        let len_1 = value[0].len();
        let len_2 = value[0][0].len();
        let len_3 = value[0][0][0].len();

        for v1 in value {
            assert_eq!(len_1, v1.len());

            for v2 in v1 {
                assert_eq!(len_2, v2.len());

                for v3 in v2 {
                    assert_eq!(len_3, v3.len());

                    for v4 in v3 {
                        data.push(v4.clone());
                    }
                }
            }
        }

        Tensor::from_vec(data, [value.len(), len_1, len_2, len_3])
    }
}

impl<T: Clone + Type> From<&Vec<Vec<Vec<Vec<Vec<T>>>>>> for Tensor<T> {
    fn from(value: &Vec<Vec<Vec<Vec<Vec<T>>>>>) -> Self {
        let len_1 = value[0].len();
        let len_2 = value[0][0].len();
        let len_3 = value[0][0][0].len();
        let len_4 = value[0][0][0][0].len();

        let data: Vec<T> = value.iter().flat_map(|v1| {
            assert_eq!(len_1, v1.len());

            v1.iter().flat_map(|v2| {
                assert_eq!(len_2, v2.len());

                v2.iter().flat_map(|v3| {
                    assert_eq!(len_3, v3.len());

                    v3.iter().flat_map(|v4| {
                        assert_eq!(len_4, v4.len());

                        v4.iter().map(|v5| v5.clone())
                    })
                })
            })
        }).collect();

        Tensor::from_vec(data, [value.len(), len_1, len_2, len_3, len_4])
    }
}

// array conversions

// can't generalize from Dtype because internal array's can't match such
// as [[1., 2.]], where T=f32 not T=[f32; 2]
impl<T: Clone + Type + 'static> From<&[T]> for Tensor<T> {
    fn from(value: &[T]) -> Self {
        let len = value.len();

        let data = Vec::from(value);

        Tensor::from_vec(data, [len])
    }
}

impl<T: Type + 'static, const N: usize> From<[T; N]> for Tensor<T> {
    fn from(value: [T; N]) -> Self {
        let vec = Vec::<T>::from(value);
        let len = vec.len();

        Tensor::from_vec(vec, Shape::from(len))
    }
}

impl<T: Clone + Type, const N: usize> From<&[[T; N]]> for Tensor<T> {
    fn from(value: &[[T; N]]) -> Self {
        let len = value.len();

        let data: Vec<T> = value.iter()
            .flat_map(|row| row.iter().map(|v| v.clone()))
            .collect();

        Tensor::from_vec(data, [len, N])
    }
}

impl From<&str> for Tensor<String> {
    fn from(value: &str) -> Self {
        Tensor::from_vec(vec![value.to_string()], Shape::scalar())
    }
}

impl From<Vec<&str>> for Tensor<String> {
    fn from(value: Vec<&str>) -> Self {
        let len = value.len();

        let vec = value.iter().map(|s| s.to_string()).collect();

        Tensor::<String>::from_vec(vec, Shape::from(len))
    }
}

impl<const N:usize> From<[&str; N]> for Tensor<String> {
    fn from(value: [&str; N]) -> Self {
        let vec : Vec<String> = value.iter().map(|s| s.to_string()).collect();
        let len = vec.len();

        Tensor::from_vec(vec, Shape::from(len))
    }
}

impl<T, const N: usize, const M: usize> From<[[T; N]; M]> for Tensor<T>
where
    T: Type,
{
    fn from(value: [[T; N]; M]) -> Self {
        let data: Vec<T> = value.into_iter()
            .flatten()
            .collect();

        Tensor::from_vec(data, [M, N])
    }
}

impl<T, const N: usize, const M: usize, const L: usize>
    From<[[[T; N]; M]; L]> for Tensor<T>
where
    T: Type
{
    fn from(value: [[[T; N]; M]; L]) -> Self {
        let data: Vec<T> = value.into_iter()
            .flatten()
            .flatten()
            .collect();

        Tensor::from_vec(data, [L, M, N])
    }
}

impl<T, const N: usize, const M: usize, const L: usize, const K: usize>
    From<[[[[T; N]; M]; L]; K]> for Tensor<T>
where
    T: Type
{
    fn from(value: [[[[T; N]; M]; L]; K]) -> Self {
        let data: Vec<T> = value.into_iter()
            .flatten()
            .flatten()
            .flatten()
            .collect();

        Tensor::from_vec(data, [K, L, M, N])
    }
}

impl<T: Type> FromIterator<T> for Tensor<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let vec : Vec<T> = Vec::from_iter(iter);
        let len = vec.len();

        Tensor::from_vec(vec, len)
    }
}

impl<'a, T: Type + Clone> FromIterator<&'a T> for Tensor<T> {
    fn from_iter<I: IntoIterator<Item = &'a T>>(iter: I) -> Self {
        let vec : Vec<T> = iter.into_iter().map(|v| v.clone()).collect();
        let len = vec.len();

        Tensor::from_vec(vec, len)
    }
}

impl<T: Type + Copy + 'static> From<&Vec<Tensor<T>>> for Tensor<T> {
    fn from(values: &Vec<Tensor<T>>) -> Self {
        Tensor::<T>::from(values.as_slice())
    }
}

impl<T: Type + Copy + 'static> From<&[Tensor<T>]> for Tensor<T> {
    fn from(values: &[Tensor<T>]) -> Self {
        let n = values.len();
        assert!(n > 0);

        let sublen = values[0].size();

        let shape = values[0].shape();

        for value in values {
            assert_eq!(sublen, value.size(), "tensor length must match");
            assert_eq!(shape, value.shape(), "tensor shapes must match");
        }

        let data : Vec<T> = values.iter().flat_map(|tensor|
            tensor.as_slice().iter().map(|v| v.clone())
        ).collect();

        Tensor::from_vec(data, shape.push(n)) 
    }
}

impl<T: Type + Copy + 'static, const N: usize> From<[Tensor<T>; N]> for Tensor<T> {
    fn from(values: [Tensor<T>; N]) -> Self {
        let vec = Vec::from(values);

        Tensor::from(&vec)
    }
}

pub trait IntoTensorList<T: Type> {
    fn into_list(self, vec: &mut Vec<Tensor<T>>);
}

impl<T: Type> IntoTensorList<T> for Vec<Tensor<T>> {
    fn into_list(self, vec: &mut Vec<Tensor<T>>) {
        let mut this = self;

        vec.append(&mut this)
    }
}

impl<T: Type> IntoTensorList<T> for Vec<&Tensor<T>> {
    fn into_list(self, vec: &mut Vec<Tensor<T>>) {
        for tensor in self {
            vec.push(tensor.clone());
        }
    }
}

impl<T: Type> IntoTensorList<T> for &[Tensor<T>] {
    fn into_list(self, vec: &mut Vec<Tensor<T>>) {
        for tensor in self {
            vec.push(tensor.clone());
        }
    }
}

impl<T: Type> IntoTensorList<T> for &[&Tensor<T>] {
    fn into_list(self, vec: &mut Vec<Tensor<T>>) {
        for tensor in self {
            vec.push((*tensor).clone());
        }
    }
}

impl<T: Type, const N: usize> IntoTensorList<T> for [Tensor<T>; N] {
    fn into_list(self, vec: &mut Vec<Tensor<T>>) {
        for tensor in self {
            vec.push(tensor);
        }
    }
}

impl<T: Type, const N: usize> IntoTensorList<T> for [&Tensor<T>; N] {
    fn into_list(self, vec: &mut Vec<Tensor<T>>) {
        for tensor in self {
            vec.push(tensor.clone());
        }
    }
}

impl<T: Type> IntoTensorList<T> for Vec<Vec<T>> {
    fn into_list(self, vec: &mut Vec<Tensor<T>>) {
        for item in self {
            vec.push(Tensor::from(item));
        }
    }
}

impl<T: Type + Clone> IntoTensorList<T> for Vec<&Vec<T>> {
    fn into_list(self, vec: &mut Vec<Tensor<T>>) {
        for item in self {
            vec.push(Tensor::<T>::from(item));
        }
    }
}

impl<T: Type + Clone> IntoTensorList<T> for &[Vec<T>] {
    fn into_list(self, vec: &mut Vec<Tensor<T>>) {
        for item in self {
            vec.push(Tensor::<T>::from(item));
        }
    }
}

impl<T: Type + Clone> IntoTensorList<T> for &[&Vec<T>] {
    fn into_list(self, vec: &mut Vec<Tensor<T>>) {
        for item in self {
            vec.push(Tensor::<T>::from(*item));
        }
    }
}

impl<T: Type, const N: usize> IntoTensorList<T> for [Vec<T>; N] {
    fn into_list(self, vec: &mut Vec<Tensor<T>>) {
        for item in self {
            vec.push(Tensor::<T>::from(item));
        }
    }
}

impl<T: Type + Clone, const N: usize> IntoTensorList<T> for [&Vec<T>; N] {
    fn into_list(self, vec: &mut Vec<Tensor<T>>) {
        for item in self {
            vec.push(Tensor::<T>::from(item));
        }
    }
}

macro_rules! into_tensor_list {
    ($($id:ident),*) => {
        #[allow(non_snake_case)]
        impl<D: Type, $($id),*> IntoTensorList<D> for ($($id,)*) 
        where $(
            $id: Into<Tensor<D>>
        ),*
        {
            fn into_list(self, vec: &mut Vec<Tensor<D>>) {
                let ($($id,)*) = self;

                $(
                    vec.push($id.into())
                );*
            }
        }
    }
}

into_tensor_list!(P0);
into_tensor_list!(P0, P1);
into_tensor_list!(P0, P1, P2);
into_tensor_list!(P0, P1, P2, P3);
into_tensor_list!(P0, P1, P2, P3, P4);
into_tensor_list!(P0, P1, P2, P3, P4, P5);
into_tensor_list!(P0, P1, P2, P3, P4, P5, P6);
into_tensor_list!(P0, P1, P2, P3, P4, P5, P6, P7);
into_tensor_list!(P0, P1, P2, P3, P4, P5, P6, P7, P8);
into_tensor_list!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9);
