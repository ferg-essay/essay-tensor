use essay_opt::derive_opt;

#[derive_opt(TestOpt)]
#[derive(Default)]
pub struct Test2 {
    name: Option<String>,
}

pub struct Test {} 

#[test]
fn test() {
    assert_eq!(test_opt(()), "None");
    assert_eq!(test_opt(().name("name-a")), "name-a");
}

fn test_opt(opt: impl TestOpt) -> String {
    let opt = opt.into_arg();

    match opt.name {
        Some(name) => format!("{}", name),
        None => format!("None"),
    }
}