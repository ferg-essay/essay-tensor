use std::{marker::PhantomData, any::{TypeId, Any, type_name}, rc::Rc, cell::RefCell};

use essay_plot_base::Coord;

use crate::frame::Data;

use super::Artist;

pub struct ArtistStyle<M: Coord> {
    artists: Vec<ArtistCell<M>>,
}

pub struct ArtistCell<M: Coord> {
    type_id: TypeId,
    artist: Box<dyn Artist<M>>,
}

pub struct Accessor<M: Coord, A: Artist<M>> {
    artist: A,
    marker: PhantomData<M>,
}

impl<M: Coord, A: Artist<M>> Accessor<M, A> {
    pub fn read<R>(&self, fun: impl FnOnce(&A) -> R) -> R {
        fun(&self.artist)
    }
}

pub struct Art {

}

impl Artist<Data> for Art {
    fn get_extent(&mut self) -> essay_plot_base::Bounds<Data> {
        todo!()
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn essay_plot_base::driver::Renderer,
        to_canvas: &essay_plot_base::Affine2d,
        clip: &essay_plot_base::Bounds<essay_plot_base::Canvas>,
        style: &dyn essay_plot_base::StyleOpt,
    ) {
        todo!()
    }
}

pub trait ArtTrait {
    fn accessor(&self, fun: &dyn FnOnce(&Self));
}

pub struct ArtAccessor<'a, M: Coord, A: Artist<M>> {
    art: Rc<RefCell<A>>,
    marker: PhantomData<&'a M>,
}

impl<M, A> ArtAccessor<'_, M, A>
where
    M: Coord,
    A: Artist<M>,
{
    pub fn new(art: Rc<RefCell<A>>) -> Self {
        Self {
            art,
            marker: PhantomData,
        }
    }
}

pub struct ArtHolder<M: Coord, A: Artist<M>> {
    art: Rc<RefCell<A>>,
    marker: PhantomData<M>,
}

impl<M, A> ArtHolder<M, A>
where
    M: Coord,
    A: Artist<M>,
{
    pub fn new(art: Rc<RefCell<A>>) -> Self {
        Self {
            art,
            marker: PhantomData,
        }
    }
}

impl<M, A> Artist<M> for ArtHolder<M, A>
where
    M: Coord,
    A: Artist<M>,
{
    fn get_extent(&mut self) -> essay_plot_base::Bounds<M> {
        todo!()
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn essay_plot_base::driver::Renderer,
        to_canvas: &essay_plot_base::Affine2d,
        clip: &essay_plot_base::Bounds<essay_plot_base::Canvas>,
        style: &dyn essay_plot_base::StyleOpt,
    ) {
        todo!()
    }
}

impl<'a, M: Coord, A: Artist<M>> ArtAccessor<'a, M, A> {
    pub fn read<R>(&self, fun: impl FnOnce(&A) -> R) -> R {
        fun(&self.art.borrow())
    }

    pub fn write<R>(&mut self, fun: impl FnOnce(&mut A) -> R) -> R {
        fun(&mut self.art.borrow_mut())
    }
}

impl<M: Coord> ArtistCell<M> {
    pub fn read<A, R>(&self, fun: &dyn FnOnce(&A) -> R) -> R
    where
        A: Artist<M>,
    {
        //assert!(TypeId::of::<M>() == self.type_id);
        let art = Art{};
        let rcart = Rc::new(RefCell::new(art));
        let acc = ArtAccessor {
            art: rcart.clone(),
            marker: PhantomData,
        };

        let holder = ArtHolder {
            art: rcart,
            marker: PhantomData,
        };

        let artbox : Box<dyn Artist<Data>> = Box::new(holder);

        todo!();
    }
}
