use encase::nalgebra::Vector2;

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
        let y = offset.y + self.radius as i64;
        let x = offset.x + self.radius as i64;
        let out = y * self.kernel_side_len() as i64 + x;
        out.try_into().unwrap()
    }
    pub fn wgsl_linear_idx_from_yx_offset(&self, offset_var: &str) -> String{
        let radius = self.radius;
        let kernel_side_len = self.kernel_side_len();

        let y = format!("({offset_var}.y + {radius})");
        let x = format!("({offset_var}.x + {radius})");
        format!("( {y} * {kernel_side_len}  + {x} )")
    }
    pub fn fill_slice_yx(&self, buffer: &mut [f32]){
        let iradius: i64 = self.radius.into();

        let mut total: f32 = 0.0;
        for y in -iradius..=iradius{
            for x in -iradius..=iradius{
                let offset = Vector2::new(x, y);
                let index = self.linear_idx_from_yx_offset(offset);
                let val = self.kernel_at(offset);
                buffer[index] = val;
                total += val;
            }
        }
        dbg!(total);
        assert!(1.0 - total < 0.001);
    }
}
