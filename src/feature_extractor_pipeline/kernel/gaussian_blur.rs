use encase::{nalgebra::{Vector2, Vector3}};

use crate::feature_extractor_pipeline::output_buffer::KernelBufferSlot;

#[derive(Clone)]
pub struct GaussianBlur {
    pub sigma: f32,
    pub radius: u8,
}

impl GaussianBlur {
    pub fn new(sigma: f32, kernel_radius: u8) -> Self{
        Self{
            sigma,
            radius: kernel_radius,
        }
    }
    pub fn kernel_side_len(&self) -> usize{
        2 * usize::from(self.radius) + 1
    }
    pub fn required_size_in_bytes(&self) -> usize{
        self.kernel_side_len().pow(2) * std::mem::size_of::<f32>()
    }
    pub fn kernel_at(&self, center_offset: Vector2<i64>) -> f32 {
        use std::f32::consts::{PI, E};
        
        let x_2 = (center_offset.x * center_offset.x) as f32;
        let y_2 = (center_offset.y * center_offset.y) as f32;
        let two_sigma_2 = 2f32 * self.sigma * self.sigma;
        let exponent = -(x_2 + y_2) / two_sigma_2;

        return (1f32 / (PI * two_sigma_2)) * E.powf(exponent);
    }
    pub fn linear_idx_from_yx_offset(&self, offset: Vector2<i64>) -> usize{
        let out = offset.y * self.kernel_side_len() as i64 + offset.x;
        out.try_into().unwrap()
    }
    pub fn fill_slice_yx(&self, buffer: &mut [f32]){
        let iradius: i64 = self.radius.into();
        let kernel_side: i64 = self.kernel_side_len().try_into().unwrap();

        let mut total: f32 = 0.0;
        for y in -iradius..=iradius{
            for x in -iradius..=iradius{
                let index = usize::try_from(y * kernel_side + x).unwrap();
                let val = self.kernel_at(Vector2::new(x, y));
                buffer[index] = val;
                total += val;
            }
        }
        assert!(1.0 - total < 0.0001);
    }
}
