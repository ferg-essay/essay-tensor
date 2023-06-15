use proc_macro::{self};
use syn::{DeriveInput, parse_macro_input, DataStruct, Result, 
    parse::{ParseStream, Parse}, 
    Fields, Ident, Type
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

pub(crate) fn derive_plot(
    attr: proc_macro::TokenStream, 
    item: proc_macro::TokenStream
) -> proc_macro::TokenStream {
    if attr.is_empty() {
        panic!("missing required attribute to #[derive_plot(Opt)]")
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

    // println!("zop {}", style_prompt(&data));
    //println!("Struct: {:?}", &struct_token.to_token_stream());
    //println!("Fields: {:?}", &fields.to_token_stream());
    //println!("Attrs: {:?}", &attrs.iter().map(|x| x.to_token_stream()));
    // struct_token.
    // println!("Item: {}", item.as);

    let trait_methods = trait_methods(&fields, &ident);
    let arg_methods = arg_methods(&fields);
    let nil_methods = nil_methods(&fields, &ident);

    //if true { return quote! {}.into(); }

    quote! {
        #(#attrs)*
        #vis #struct_token #ident #generics 
            #fields
        #semi_token

        #vis trait #opt {
            #trait_methods

            fn into_arg(self) -> #ident;
        }

        impl #opt for #ident {
            #arg_methods

            fn into_arg(self) -> Self {
                self
            }
        }

        impl #opt for () {
            #nil_methods

            fn into_arg(self) -> #ident {
                #ident::default()
            }
        }
    }.into()
}

fn trait_methods(fields: &Fields, arg: &Ident) -> TokenStream {
    match fields {
        Fields::Named(ref fields) => {
            let iter = fields.named.iter().map(|field| {
                let name = &field.ident;
                let ty = &field.ty;

                match option_type(ty) {
                    Some(ty) => {
                        if is_into_type(&ty) {
                            quote! {
                                fn #name(self, #name: impl Into<#ty>) -> #arg;
                            }
                        } else {
                            quote! {
                                fn #name(self, #name: #ty) -> #arg;
                            }
                        }
                    },
                    None => {
                        quote! {
                            fn #name(self, value: #ty) -> #arg;
                        }
                    }
                }
                
            });

            quote! { #(#iter)* }
        }
        _ => quote! {}
    }
}

fn arg_methods(fields: &Fields) -> TokenStream {
    match fields {
        Fields::Named(ref fields) => {
            let iter = fields.named.iter().map(|field| {
                let name = &field.ident;
                let ty = &field.ty;

                match option_type(ty) {
                    Some(ty) => {
                        if is_into_type(&ty) {
                            quote! {
                                fn #name(self, #name: impl Into<#ty>) -> Self {
                                    Self { #name: Some(#name.into()), ..self }
                                }
                            }
                        } else {
                            quote! {
                                fn #name(self, #name: #ty) -> Self {
                                    Self { #name: Some(#name), ..self }
                                }
                            }
                        }
                    },
                    None => {
                        quote! {
                            fn #name(self, #name: #ty) -> Self {
                                Self { #name: #name, ..self }
                            }
                        }
                    }
                }

            });

            quote! { #(#iter)* }
        }
        _ => quote! {}
    }
}

fn is_into_type(ty: &Type) -> bool {
    if let syn::Type::Path(syn::TypePath { qself: None, path }) = ty {
        let path = path_to_string(&path);

        if path == "usize" {
            return false;
        }
    }

    true
}

fn nil_methods(fields: &Fields, arg: &Ident) -> TokenStream {
    match fields {
        Fields::Named(ref fields) => {
            let iter = fields.named.iter().map(|field| {
                let name = &field.ident;
                let ty = &field.ty;

                match option_type(ty) {
                    Some(ty) => {
                        if is_into_type(&ty) {
                            quote! {
                                fn #name(self, #name: impl Into<#ty>) -> #arg {
                                    #arg::default().#name(#name)
                                }
                            }
                        } else {
                            quote! {
                                fn #name(self, #name: #ty) -> #arg {
                                    #arg::default().#name(#name)
                                }
                            }
                        }
                    },
                    None => {
                        quote! {
                            fn #name(self, #name: #ty) -> #arg {
                                #arg::default().#name(#name)
                            }
                        }
                    }
                }
            });

            quote! { #(#iter)* }
        }
        _ => quote! {}
    }
}

fn option_type(ty: &Type) -> Option<Type> {
    if let syn::Type::Path(syn::TypePath { qself: None, path }) = ty {
        if path_to_string(&path) == "Option" {
            if let Some(tail) = path.segments.last() {
                match &tail.arguments {
                    syn::PathArguments::AngleBracketed(syn::AngleBracketedGenericArguments {
                        args,
                        ..
                    }) => {
                        if let Some(syn::GenericArgument::Type(ty)) = args.first() {
                            return Some(ty.clone())
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    None
}

fn path_to_string(path: &syn::Path) -> String {
    path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>()
        .join(":")
}
