use proc_macro::TokenStream;

mod plot_opt;
//mod canvas_opt;

#[proc_macro_attribute]
pub fn derive_plot_opt(attr: TokenStream, item: TokenStream) -> TokenStream {
    plot_opt::derive_plot_opt(attr, item)
}
/*
#[proc_macro_attribute]
pub fn derive_canvas_opt(attr: TokenStream, item: TokenStream) -> TokenStream {
    plot_opt::derive_canvas_opt(attr, item)
}
*/
