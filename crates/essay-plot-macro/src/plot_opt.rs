use proc_macro::{self};
use syn::{DeriveInput, parse_macro_input, DataStruct, Result, 
    parse::{ParseStream, Parse}, 
    Fields, Attribute, Field, Ident
};
use quote::{*, __private::TokenStream};

struct OptAttribute {
    opt: syn::Ident,
}

impl Parse for OptAttribute {
    fn parse(input: ParseStream) -> Result<Self> {
        let opt : syn::Ident = input.parse().unwrap();
        Ok(Self {
            opt,
        })
    }
}

pub(crate) fn derive_plot_opt(
    attr: proc_macro::TokenStream, 
    item: proc_macro::TokenStream
) -> proc_macro::TokenStream {
    if attr.is_empty() {
        panic!("missing required attribute to #[derive_plot_opt(Opt)]")
    }

    let OptAttribute {
        opt,
    } = parse_macro_input!(attr as OptAttribute);

    let DeriveInput {
        ident,
        data,
        generics,
        attrs,
        vis,
    } = parse_macro_input!(item as DeriveInput);

    let DataStruct {
        struct_token,
        fields,
        semi_token,
    } = match data {
        syn::Data::Struct(data) => data,
        syn::Data::Enum(_) => todo!("enum not supported for derive_opt"),
        syn::Data::Union(_) => todo!("union not supported for derive_opt"),
    };

    let struct_fields = struct_fields(&fields);
    let field_methods = field_methods(&fields);

    quote! {
        #(#attrs)*
        #vis #struct_token #ident #generics 
            #struct_fields
        #semi_token

        #vis struct #opt {
            plot: essay_plot::graph::PlotRef<essay_plot::frame::Data, #ident>,
        }

        impl #opt {
            pub fn new(
                plot: essay_plot::graph::PlotRef<essay_plot::frame::Data, #ident>,
            ) -> Self {
                Self {
                    plot
                }
            }

            #field_methods
        }
    }.into()
}

fn struct_fields(fields: &Fields) -> TokenStream {
    match fields {
        Fields::Named(ref fields) => {
            let iter = fields.named.iter().map(|field| {
                let ident = &field.ident;
                let ty = &field.ty;
                let vis = &field.vis;
                let attrs = print_opt(&field.attrs);

                quote! {
                    #attrs
                    #vis #ident: #ty,
                }
            });

            quote! { { #(#iter)* } }
        }
        // _ => quote! {}
        _ => todo!()
    }
}

fn field_methods(fields: &Fields) -> TokenStream {
    match fields {
        Fields::Named(ref fields) => {
            let iter = fields.named.iter().map(|field| {
                let name = &field.ident;
                let ty = &field.ty;

                if is_opt(&field.attrs) {
                    if is_opt_into(&field.attrs) {
                        quote! {
                            pub fn #name(&mut self, #name: impl Into<#ty>) -> &mut Self {
                                self.plot.write_artist(|a| { a.#name(#name); });

                                self
                            }
                        }
                    } else if is_path_opt(&field) {
                        path_opt_methods(name)
                    } else {
                        quote! {
                            pub fn #name(&mut self, #name: #ty) -> &mut Self {
                                self.plot.write_artist(|a| { a.#name(#name); });

                                self
                            }
                        }
                    }
                } else {
                    quote! {}
                }

            });

            quote! { #(#iter)* }
        }
        _ => quote! {}
    }
}

fn path_opt_methods(name: &Option<Ident>) -> TokenStream {
    quote! {
        /// Sets the path's fill_color and edge_color
        pub fn color(
            &mut self, 
            color: impl Into<essay_plot_base::Color>
        ) -> &mut Self {
            self.plot.write_artist(|a| { a.#name.color(color); });
            self
        }

        /// Sets the path's fill color, which is used for both filling
        /// and as a default line color.
        pub fn face_color(
            &mut self, 
            color: impl Into<essay_plot_base::Color>
        ) -> &mut Self {
            self.plot.write_artist(|a| { a.#name.face_color(color); });
            self
        }

        /// Sets the path's outline edge color when it's distinct from the fill color.
        pub fn edge_color(
            &mut self, 
            color: impl Into<essay_plot_base::Color>
        ) -> &mut Self {
            self.plot.write_artist(|a| { a.#name.edge_color(color); });
            self
        }

        /// Sets the line style to Solid, Dashed, DashDot, etc.
        pub fn line_style(
            &mut self, 
            style: impl Into<essay_plot_base::LineStyle>
        ) -> &mut Self {
            self.plot.write_artist(|a| { a.#name.line_style(style); });
            self
        }

        /// Sets the path's line width in logical pixels.
        pub fn line_width(
            &mut self, 
            width: f32,
        ) -> &mut Self {
            self.plot.write_artist(|a| { a.#name.line_width(width); });
            self
        }

        /// Sets the path's style for joining two line 
        /// segments. One of Bevel, Miter, or Round.
        pub fn join_style(
            &mut self, 
            style: impl Into<essay_plot_base::JoinStyle>,
        ) -> &mut Self {
            self.plot.write_artist(|a| { a.#name.join_style(style); });
            self
        }

        /// Sets the path's style for the end of a line segment.
        /// One of Butt, Projecting, or Round.
        pub fn cap_style(
            &mut self, 
            style: impl Into<essay_plot_base::CapStyle>,
        ) -> &mut Self {
            self.plot.write_artist(|a| { a.#name.cap_style(style); });
            self
        }

    }
}

fn is_opt(attrs: &Vec<Attribute>) -> bool {
    for attr in attrs {
        if attr.path().is_ident("option") {
            return true;
        }
    }

    false
}

fn is_path_opt(field: &Field) -> bool {
    match &field.ty {
        syn::Type::Path(path) => {
            path.path.is_ident("PathStyle")
        },
        _ => false,
    }
}

fn is_opt_into(attrs: &Vec<Attribute>) -> bool {
    let mut is_into = false;

    for attr in attrs {
        if attr.path().is_ident("option") {
            match &attr.meta {
                syn::Meta::Path(_path) => { return false; }
                syn::Meta::List(_list) => {
                    attr.parse_nested_meta(|meta| {
                        if meta.path.is_ident("Into") {
                            is_into = true;
                        } else {
                            panic!("Expected 'option(Into)'");
                        }
                        Ok(())
                    }).unwrap();
                }
                syn::Meta::NameValue(_) => todo!("unexpected name-value"),
            }
        }
    }

    is_into
}

fn print_opt(attrs: &Vec<Attribute>) -> TokenStream {
    let iter = attrs.iter().filter(|a| ! a.path().is_ident("option"));

    quote! { #(#iter)* }
}
