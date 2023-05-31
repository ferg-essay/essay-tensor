use proc_macro::TokenStream;

mod options;

#[proc_macro_attribute]
pub fn derive_opt(attr: TokenStream, item: TokenStream) -> TokenStream {
    options::derive_opt(attr, item)
}
