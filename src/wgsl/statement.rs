use crate::wgsl::{Wgsl, ShaderTypeExt};
use crate::wgsl::declaration::FunctionVarDecl;
use crate::wgsl::expression::Expression;

use super::buffer::OutputBufferDecl;

pub trait Statement: Wgsl{}

macro_rules! declare_assigment {
    ($name:ident, $operator:literal) => {
        pub struct $name<T: ShaderTypeExt>{
            pub assignee: FunctionVarDecl<T>,
            pub value: Expression<T>,
        }
        impl<T: ShaderTypeExt> Wgsl for $name<T>{
            fn wgsl(&self) -> String {
                let assignee_wgsl = &self.assignee.name;
                let val_wgsl = self.value.wgsl();
                format!("{assignee_wgsl} {} {val_wgsl};", $operator)
            }
        }
        impl<T: ShaderTypeExt> Statement for $name<T>{}
    };
}

declare_assigment!(Assignment, "=");
declare_assigment!(AddAssignment, "+=");


pub struct BufferWrite<T: ShaderTypeExt>{
    pub buffer: OutputBufferDecl<T>,
    pub index: Expression<u32>,
    pub value: Expression<T>,
}
impl<T: ShaderTypeExt> Wgsl for BufferWrite<T>{
    fn wgsl(&self) -> String {
        let Self{buffer, index, value} = self;
        let buffer_name = &buffer.name;
        format!("{buffer_name}[{index}] = {value};")
    }
}
impl<T: ShaderTypeExt> Statement for BufferWrite<T>{}
