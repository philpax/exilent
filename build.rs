use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

fn main() {
    println!("cargo:rerun-if-changed=resource");

    let files: HashMap<PathBuf, String> = walkdir::WalkDir::new("resource")
        .into_iter()
        .filter_map(Result::ok)
        .map(|d| d.into_path())
        .filter(|p| p.is_file())
        .map(|p| {
            (
                p.strip_prefix("resource").unwrap().to_owned(),
                hex::encode(std::fs::read(p).unwrap()),
            )
        })
        .collect();

    std::fs::write(
        Path::new(&std::env::var("OUT_DIR").unwrap()).join("resource.json"),
        serde_json::to_string(&files).unwrap(),
    )
    .unwrap();
}
