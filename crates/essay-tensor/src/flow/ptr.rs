use std::{any::TypeId, ptr::NonNull, alloc::Layout, mem::{ManuallyDrop, self}};


pub(crate) struct Ptr {
    type_id: TypeId,
    data: NonNull<u8>,
}

impl Ptr {
    pub(crate) unsafe fn wrap<T: 'static>(value: T) -> Self {
        let type_id = TypeId::of::<T>();

        let layout = Layout::for_value(&value);

        let data = std::alloc::alloc(layout);
        let data = NonNull::new(data).unwrap();

        // TODO: drop
        let mut value = ManuallyDrop::new(value);
        let source = NonNull::from(&mut *value).cast();

        std::ptr::copy_nonoverlapping::<u8>(
            source.as_ptr(), 
            data.as_ptr(),
            mem::size_of::<T>(),
        );

        Self {
            type_id: TypeId::of::<T>(),
            data,
        }
    }

    pub(crate) unsafe fn as_ref<T: 'static>(&self) -> &T {
        assert_eq!(self.type_id, TypeId::of::<T>());

        self.data.cast::<T>().as_ref()
    }

    pub(crate) unsafe fn as_mut<T: 'static>(&mut self) -> &mut T {
        assert_eq!(self.type_id, TypeId::of::<T>());

        self.data.cast::<T>().as_mut()
    }

    pub(crate) unsafe fn unwrap<T: 'static>(self) -> T {
        assert_eq!(self.type_id, TypeId::of::<T>());

        self.data.cast::<T>().as_ptr().read()
    }
}
