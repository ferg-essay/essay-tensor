use proc_macro::{self};
use syn::{DeriveInput, parse_macro_input, DataStruct, Result, 
    parse::{ParseStream, Parse}, 
    Fields, Ident, Type, Attribute
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

    //Attribute::parse_inner(ParseStream::call(attr));

    //let opt = attr.into_iter().next();
    //let attr: syn::Attribute = parse_macro_input!(attr);
    //let attr = parse_macro_input!(attr as Attribute);
    // let attr = attr.into_iter().next();
    
    //let opt : Ident = syn::parse(attr).unwrap();
    
    //println!("Attr {:?}", opt.to_token_stream());

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

    //println!("Struct: {:?}", &struct_token.to_token_stream());
    //println!("Fields: {:?}", &fields.to_token_stream());
    //println!("Attrs: {:?}", &attrs.iter().map(|x| x.to_token_stream()));
    // struct_token.
    // println!("Item: {}", item.as);

    //let trait_methods = trait_methods(&fields, &ident);
    let struct_fields = struct_fields(&fields);
    let field_methods = field_methods(&fields);

    //if true { return quote! {}.into(); }

    quote! {
        #(#attrs)*
        #vis #struct_token #ident #generics 
            #struct_fields
        #semi_token

        #vis struct #opt {
            plot: crate::graph::PlotRef<crate::frame::Data, #ident>,
        }

        impl #opt {
            pub fn new(
                plot: crate::graph::PlotRef<crate::frame::Data, #ident>,
            ) -> Self {
                Self {
                    plot
                }
            }

            #field_methods

            pub fn edgecolor(
                &mut self,
                color: impl Into<essay_plot_base::Color>
            ) -> &mut Self {
                self.plot.write_style(|s| { s.edgecolor(color); } );
                self
            }

            pub fn facecolor(
                &mut self,
                color: impl Into<essay_plot_base::Color>
            ) -> &mut Self {
                self.plot.write_style(|s| { s.facecolor(color); } );
                self
            }

            pub fn color(
                &mut self,
                color: impl Into<essay_plot_base::Color>
            ) -> &mut Self {
                self.plot.write_style(|s| { s.color(color); } );
                self
            }
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
                // let generics = &field.generics;
                println!("Attrs {:?}", field.attrs.len());
                //println!("Ty {:?}", ty.to_token_stream());

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
                println!("Attrs {:?}", field.attrs.len());
                //println!("Ty {:?}", ty.to_token_stream());

                if is_opt(&field.attrs) {
                    if is_opt_into(&field.attrs) {
                        quote! {
                            pub fn #name(&mut self, #name: impl Into<#ty>) -> &mut Self {
                                self.plot.write_artist(|a| { a.#name(#name); });

                                self
                            }
                        }
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

fn is_opt(attrs: &Vec<Attribute>) -> bool {
    for attr in attrs {
        if attr.path().is_ident("option") {
            return true;
        }
    }

    false
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

fn is_into_type(_ty: &Type) -> bool {
    /*
    if let syn::Type::Path(syn::TypePath { qself: None, path }) = ty {
        let path = path_to_string(&path);

        if path == "usize" {
            return false;
        }
    }

    true
    */
    false
}

fn path_to_string(path: &syn::Path) -> String {
    path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>()
        .join(":")
}
