use crate::wgsl::{Wgsl, ShaderTypeExt};
use crate::wgsl::declaration::FunctionVarDecl;
use crate::wgsl::expression::Expression;

use super::buffer::OutputBufferDecl;

pub trait Statement: Wgsl{}

macro_rules! declare_assigment {
    ($name:ident, $operator:literal) => {
        pub struct $name<T: ShaderTypeExt>{
            assignee: FunctionVarDecl<T>,
            value: Expression<T>,
        }
        impl<T: ShaderTypeExt> Wgsl for $name<T>{
            fn wgsl(&self) -> String {
                let assignee_wgsl = self.assignee.wgsl();
                let val_wgsl = self.value.wgsl();
                format!("{assignee_wgsl} {} {val_wgsl}", $operator)
            }
        }
        impl<T: ShaderTypeExt> Statement for $name<T>{}
    };
}

declare_assigment!(Assignment, "=");
declare_assigment!(AddAssignment, "+=");


pub struct BufferWrite<T: ShaderTypeExt>{
    buffer: OutputBufferDecl<T>,
    index: Expression<u32>,
    value: Expression<T>,
}
impl<T: ShaderTypeExt> Wgsl for BufferWrite<T>{
    fn wgsl(&self) -> String {
        let Self{buffer, index, value} = self;
        let buffer_name = &buffer.name;
        format!("{buffer_name}[{index}] = {value};")
    }
}
impl<T: ShaderTypeExt> Statement for BufferWrite<T>{}
