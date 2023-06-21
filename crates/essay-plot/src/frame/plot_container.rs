use std::{alloc, any::TypeId, marker::PhantomData, ptr::{NonNull, self}, mem::{ManuallyDrop, self}};

use essay_plot_base::{Coord, Bounds, driver::Renderer, Affine2d, Canvas, PathOpt, Clip};

use crate::{artist::{Artist, StyleCycle, PlotArtist}, graph::Config};

use super::{ArtistId, FrameId, legend::LegendHandler};

pub(crate) struct PlotContainer<M: Coord> {
    frame: FrameId,
    ptrs: Vec<PlotPtr<M>>,
    artists: Vec<Box<dyn ArtistHandleTrait<M>>>,
    cycle: StyleCycle,
}

impl<M: Coord> PlotContainer<M> {
    pub(crate) fn new(frame: FrameId, cfg: &Config) -> Self {
        let container = Self {
            frame,
            ptrs: Vec::new(),
            artists: Vec::new(),
            cycle: StyleCycle::from_config(cfg, "frame.cycle"),
        };

        container
    }

    pub(crate) fn add_artist<A>(&mut self, artist: A) -> ArtistId
    where
        A: PlotArtist<M> + 'static
    {
        let id = ArtistId::new_data(self.frame, self.ptrs.len());

        let plot = PlotPtr::new(id, artist);
        self.ptrs.push(plot);

        self.artists.push(Box::new(ArtistHandle::<M, A>::new(id)));

        id
    }

    pub(crate) fn cycle(&self) -> &StyleCycle {
        &self.cycle
    }

    pub(crate) fn cycle_mut(&mut self) -> &mut StyleCycle {
        &mut self.cycle
    }

    fn _deref<A: Artist<M> + 'static>(&self, id: ArtistId) -> &A {
        unsafe { self.ptrs[id.index()].deref() }
    }

    fn deref_mut<A: Artist<M> + 'static>(&self, id: ArtistId) -> &mut A {
        unsafe { self.ptrs[id.index()].deref_mut() }
    }

    //pub(crate) fn style_mut(&mut self, id: ArtistId) -> &mut PathStyle {
    //    self.artists[id.index()].style_mut()
    //}

    pub(crate) fn artist<A>(&self, id: ArtistId) -> &A
    where
        A: Artist<M> + 'static
    {
        unsafe { self.ptrs[id.index()].deref() }
    }

    pub(crate) fn artist_mut<A>(&mut self, id: ArtistId) -> &mut A
    where
        A: Artist<M> + 'static
    {
        unsafe { self.ptrs[id.index()].deref_mut() }
    }

    pub(crate) fn get_handlers(&self) -> Vec<LegendHandler> {
        let mut vec = Vec::<LegendHandler>::new();

        for artist in &self.artists {
            match artist.get_legend(self) {
                Some(handler) => vec.push(handler),
                None => {},
            };
        }

        vec
    }
}

impl<M: Coord> Artist<M> for PlotContainer<M> {
    fn update(&mut self, canvas: &Canvas) {
        for artist in &self.artists {
            artist.update(self, canvas);
        }
    }
    
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
        renderer: &mut dyn Renderer,
        to_canvas: &Affine2d,
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        for (i, artist) in self.artists.iter().enumerate() {
            let style = self.cycle.push(style, i);

            artist.draw(self, renderer, to_canvas, clip, &style);
        }
    }
}

trait ArtistHandleTrait<M: Coord> {
    fn id(&self) -> ArtistId;

    //fn style_mut(&mut self) -> &mut PathStyle;

    fn update(&self, container: &PlotContainer<M>, canvas: &Canvas);
    fn get_extent(&self, container: &PlotContainer<M>) -> Bounds<M>;
    fn get_legend(&self, container: &PlotContainer<M>) -> Option<LegendHandler>;

    fn draw(
        &self, 
        container: &PlotContainer<M>,
        renderer: &mut dyn Renderer,
        to_canvas: &Affine2d,
        clip: &Clip,
        style: &dyn PathOpt,
    );
}

struct ArtistHandle<M: Coord, A: Artist<M>> {
    id: ArtistId,
    marker: PhantomData<(M, A)>,
}

impl<M: Coord, A: Artist<M>> ArtistHandle<M, A> {
    fn new(id: ArtistId) -> Self {
        Self {
            id,
            marker: PhantomData,
        }
    }
}

impl<M: Coord, A: Artist<M>> ArtistHandleTrait<M> for ArtistHandle<M, A>
where
    M: Coord,
    A: PlotArtist<M> + 'static,
{
    fn id(&self) -> ArtistId {
        self.id
    }

    //fn style_mut(&mut self) -> &mut PathStyle {
    //    &mut self.style
    //}

    fn update(&self, container: &PlotContainer<M>, canvas: &Canvas) {
        container.deref_mut::<A>(self.id).update(canvas);
    }

    fn get_extent(&self, container: &PlotContainer<M>) -> Bounds<M> {
        container.deref_mut::<A>(self.id).get_extent()
    }

    fn draw(
        &self, 
        container: &PlotContainer<M>,
        renderer: &mut dyn Renderer,
        to_canvas: &Affine2d,
        clip: &Clip,
        style: &dyn PathOpt,
    ) {
        container.deref_mut::<A>(self.id).draw(renderer, to_canvas, clip, style)
    }

    fn get_legend(&self, container: &PlotContainer<M>) -> Option<LegendHandler> {
        container.deref_mut::<A>(self.id).get_legend()
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
