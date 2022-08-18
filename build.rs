use std::env;

fn try_link_mkl() {
    // // pkg-config --cflags --libs mkl-dynamic-lp64-iomp
    /*if let Ok(_) = env::var("CARGO_FEATURE_MKL") {
        println!("cargo:rustc-link-lib=mkl_intel_lp64");
        println!("cargo:rustc-link-lib=mkl_intel_thread");
        println!("cargo:rustc-link-lib=mkl_core");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=iomp5");
    }*/

    // pkg-config --libs --cflags mkl-dynamic-ilp64-seq
    if let Ok(_) = env::var("CARGO_FEATURE_MKL") {
        println!("cargo:rustc-link-lib=mkl_intel_lp64");
        println!("cargo:rustc-link-lib=mkl_sequential");
        println!("cargo:rustc-link-lib=mkl_core");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=dl");
    }

}

fn try_link_ipp() {
    // NOTE: /opt/intel/oneapi/ipp/latest/lib/intel64 should be in LD_LIBRARY_PATH, because the DLLs
    // will search be searched for the linked code as well (not only by cargo).
    if let Ok(_) = env::var("CARGO_FEATURE_IPP") {
        println!("cargo:rustc-link-search=/opt/intel/oneapi/ipp/latest/lib/intel64");
        println!("cargo:rustc-link-lib=ippcore");
        println!("cargo:rustc-link-lib=ippvm");
        println!("cargo:rustc-link-lib=ipps");
        println!("cargo:rustc-link-lib=ippi");
    }
}

/*fn try_link_gsl() {
    if let Ok(_) = env::var("CARGO_FEATURE_GSL") {
        println!("cargo:rustc-link-lib=gsl");
        println!("cargo:rustc-link-lib=gslcblas");
        println!("cargo:rustc-link-lib=m");
    }
}*/

fn main() {
    try_link_mkl();
    try_link_ipp();
}

