use proc_macro::TokenStream;

mod plot_opt;

#[proc_macro_attribute]
pub fn derive_plot(attr: TokenStream, item: TokenStream) -> TokenStream {
    plot_opt::derive_plot(attr, item)
}
