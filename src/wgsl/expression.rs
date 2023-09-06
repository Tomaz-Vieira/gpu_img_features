use std::{fmt::Display, marker::PhantomData, ops::{Add, Mul}};

use nalgebra::Vector3;
use paste::paste;

use super::{IVec2, ShaderTypeExt, Wgsl, FVec4};

#[derive(Clone)]
pub struct Expression<T: ShaderTypeExt>(pub String, pub PhantomData<T>);

impl<T: ShaderTypeExt> Expression<T>{
    fn new(code: String) -> Self{
        return Self(code, PhantomData)
    }
}

impl<T: ShaderTypeExt> Wgsl for Expression<T>{
    fn wgsl(&self) -> String {
        self.0.clone()
    }
}

impl<T: ShaderTypeExt> Display for Expression<T>{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}


macro_rules! impl_Op_for_Expression {($op_name:ident, $op_symbol:literal) => {paste!{
    impl<T, RHS> std::ops::$op_name<RHS> for Expression<T>
    where
        T: ShaderTypeExt,
        RHS: Into<Expression<T>>
    {
        type Output = Expression<T>;
        fn [<$op_name:lower>](self, rhs: RHS) -> Self::Output {
            Expression::new(
                format!("({} {} {})", self.0, stringify!($op_symbol), rhs.into())
            )
        }
    }
}};}
impl_Op_for_Expression!(Add, "+");
impl_Op_for_Expression!(Sub, "-");
impl_Op_for_Expression!(Mul, "*");


impl From<f32> for Expression<f32>{
    fn from(value: f32) -> Self {
        return Expression::new(format!("{value:.20e}"))
    }
}
impl From<u32> for Expression<u32>{
    fn from(value: u32) -> Self {
        return Expression::new(format!("{value}"))
    }
}
impl From<i32> for Expression<i32>{
    fn from(value: i32) -> Self {
        return Expression::new(format!("{value}"))
    }
}
impl From<IVec2> for Expression<IVec2>{
    fn from(value: IVec2) -> Self {
        let type_name = IVec2::wgsl_type_name();
        return Expression::new(format!(
            "{type_name}({}, {})",
            value.x, value.y
        ))
    }
}
impl From<FVec4> for Expression<FVec4>{
    fn from(value: FVec4) -> Self {
        let type_name = FVec4::wgsl_type_name();
        return Expression::new(format!(
            "{type_name}({}, {}, {}, {})",
            value.x, value.y, value.z, value.w
        ))
    }
}
impl Mul<&Expression<FVec4>> for Expression<f32>{
    type Output = Expression<FVec4>;
    fn mul(self, rhs: &Expression<FVec4>) -> Self::Output {
        return Expression(format!("({self} * {rhs})"), PhantomData)
    }
}

macro_rules! impl_vec3_xyz {
    ($item_type:ty) => {
        impl Expression<Vector3<$item_type>>{
            pub fn x(&self) ->Expression<$item_type>{
                Expression::new(format!("{self}.x"))
            }
            pub fn y(&self) ->Expression<$item_type>{
                Expression::new(format!("{self}.y"))
            }
            pub fn z(&self) ->Expression<$item_type>{
                Expression::new(format!("{self}.z"))
            }
        }
    };
}
impl_vec3_xyz!(f32);
impl_vec3_xyz!(u32);

impl Expression<i32>{
    pub fn clamped(&self, min: impl Into<Expression<i32>>, max: impl Into<Expression<i32>>) -> Expression<i32>{
        let min_exp = min.into();
        let max_exp = max.into();
        return Self(format!("clamp({self}, {min_exp}, {max_exp})"), PhantomData)
    }
}

impl Expression<IVec2>{
    pub fn construct(x: impl Into<Expression<i32>>, y: impl Into<Expression<i32>>) -> Self{
        return Self(
            format!("{}({}, {})", IVec2::wgsl_type_name(), x.into(), y.into()),
            PhantomData
        )
    }
}
