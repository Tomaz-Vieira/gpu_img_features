use std::marker::PhantomData;

use crate::util::{Binding, Group};

use super::{ShaderTypeExt, Wgsl};

pub trait OutputBuffer: Wgsl {}

#[derive(Clone)]
pub struct OutputBufferDecl<T: ShaderTypeExt> {
    pub group: Group,
    pub binding: Binding,
    pub name: String,
    pub marker: PhantomData<T>,
}
impl<T: ShaderTypeExt> Wgsl for OutputBufferDecl<T> {
    fn wgsl(&self) -> String {
        let Self {
            group, binding, name, ..
        } = &self;
        let item_type_str = T::wgsl_type_name();
        // let {group, binding, ..} = self;
        format!("@group({group}) @binding({binding}) var<storage, write> {name} : array<{item_type_str}>;")
    }
}
impl<T: ShaderTypeExt> OutputBuffer for OutputBufferDecl<T> {}
