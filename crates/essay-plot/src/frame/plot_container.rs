use std::{alloc, any::TypeId, marker::PhantomData, ptr::{NonNull, self}, mem::{ManuallyDrop, self}};

use essay_plot_base::{Coord, Bounds, Style, driver::Renderer, Affine2d, Canvas, StyleOpt, style::Chain};

use crate::artist::Artist;

use super::ArtistId;

pub(crate) struct PlotContainer<M: Coord> {
    ptrs: Vec<PlotPtr<M>>,
    artists: Vec<Box<dyn PlotArtistTrait<M>>>,
}

impl<M: Coord> PlotContainer<M> {
    pub(crate) fn new() -> Self {
        Self {
            ptrs: Vec::new(),
            artists: Vec::new(),
        }
    }

    pub(crate) fn add_artist<A>(&mut self, plot: A) -> ArtistId
    where
        A: Artist<M> + 'static
    {
        let id = ArtistId(self.ptrs.len());

        let plot = PlotPtr::new(id, plot);
        self.ptrs.push(plot);

        self.artists.push(Box::new(PlotArtist::<M, A>::new(id)));

        id
    }

    fn _deref<A: Artist<M> + 'static>(&self, id: ArtistId) -> &A {
        unsafe { self.ptrs[id.index()].deref() }
    }

    fn deref_mut<A: Artist<M> + 'static>(&self, id: ArtistId) -> &mut A {
        unsafe { self.ptrs[id.index()].deref_mut() }
    }

    pub(crate) fn style_mut(&mut self, id: ArtistId) -> &mut Style {
        self.artists[id.index()].style_mut()
    }
}

impl<M: Coord> Artist<M> for PlotContainer<M> {
    fn get_extent(&mut self) -> Bounds<M> {
        let mut bounds = Bounds::none();

        for artist in &self.artists {
            bounds = if bounds.is_none() {
                artist.get_extent(self)
            } else {
                bounds.union(&artist.get_extent(self))
            }
        }

        bounds
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn essay_plot_base::driver::Renderer,
        to_canvas: &essay_plot_base::Affine2d,
        clip: &Bounds<essay_plot_base::Canvas>,
        style: &dyn essay_plot_base::StyleOpt,
    ) {
        for artist in &self.artists {
            artist.draw(self, renderer, to_canvas, clip, style);
        }
    }
}

trait PlotArtistTrait<M: Coord> {
    fn style_mut(&mut self) -> &mut Style;

    fn get_extent(&self, container: &PlotContainer<M>) -> Bounds<M>;
    fn draw(
        &self, 
        container: &PlotContainer<M>,
        renderer: &mut dyn Renderer,
        to_canvas: &Affine2d,
        clip: &Bounds<Canvas>,
        style: &dyn StyleOpt,
    );
}

struct PlotArtist<M: Coord, A: Artist<M>> {
    id: ArtistId,
    marker: PhantomData<(M, A)>,
    style: Style,
}

impl<M: Coord, A: Artist<M>> PlotArtist<M, A> {
    fn new(id: ArtistId) -> Self {
        Self {
            id,
            marker: PhantomData,
            style: Style::new(),
        }
    }

    pub(crate) fn style(&self) -> &Style {
        &self.style
    }
}

impl<M: Coord, A: Artist<M>> PlotArtistTrait<M> for PlotArtist<M, A>
where
    M: Coord,
    A: Artist<M> + 'static,
{
    fn style_mut(&mut self) -> &mut Style {
        &mut self.style
    }

    fn get_extent(&self, container: &PlotContainer<M>) -> Bounds<M> {
        container.deref_mut::<A>(self.id).get_extent()
    }

    fn draw(
        &self, 
        container: &PlotContainer<M>,
        renderer: &mut dyn essay_plot_base::driver::Renderer,
        to_canvas: &essay_plot_base::Affine2d,
        clip: &Bounds<essay_plot_base::Canvas>,
        style: &dyn essay_plot_base::StyleOpt,
    ) {
        let style = Chain::new(style, &self.style);

        container.deref_mut::<A>(self.id).draw(renderer, to_canvas, clip, &style)
    }
}

pub(crate) struct PlotPtr<M: Coord> {
    id: ArtistId,
    type_id: TypeId, 
    marker: PhantomData<M>,
    data: NonNull<u8>,
}

impl<M: Coord> PlotPtr<M> {
    pub(crate) fn new<A>(id: ArtistId, artist: A) -> Self
    where
        A: Artist<M> + 'static
    {
        let layout = alloc::Layout::new::<A>();
        let data = unsafe { alloc::alloc(layout) };
        let mut value = ManuallyDrop::new(artist);
        let source: NonNull<u8> = NonNull::from(&mut *value).cast();

        let src = source.as_ptr();
        let count = mem::size_of::<A>();

        // TODO: drop

        unsafe {
            ptr::copy_nonoverlapping::<u8>(src, data, count);
        }

        Self {
            id,
            type_id: TypeId::of::<A>(),
            data: NonNull::new(data).unwrap(),
            marker: PhantomData,
        }
    }

    pub unsafe fn deref<A>(&self) -> &A
    where
        A: Artist<M> + 'static
    {
        assert_eq!(self.type_id, TypeId::of::<A>());

        &*self.data.as_ptr().cast::<A>()
    }

    pub unsafe fn deref_mut<A>(&self) -> &mut A 
    where
        A: Artist<M> + 'static
    {
        assert_eq!(self.type_id, TypeId::of::<A>());

        &mut *self.data.as_ptr().cast::<A>()
    }
}
