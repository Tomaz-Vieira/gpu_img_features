use encase::nalgebra::Vector3;

pub struct GaussianBlur {
    pub sigma: f32,
}

impl GaussianBlur {
    pub fn kernel_at(&self, center_offset: Vector3<i32>) -> f32 {
        use std::f32::consts::{PI, E};
        
        let x_2 = (center_offset.x * center_offset.x) as f32;
        let y_2 = (center_offset.y * center_offset.y) as f32;
        let two_sigma_2 = 2f32 * self.sigma * self.sigma;
        let exponent = -(x_2 + y_2) / two_sigma_2;

        return (1f32 / (PI * two_sigma_2)) * E.powf(exponent);
    }
}

#[test]
fn test_gaussian() {
    let gb = GaussianBlur {
        sigma: 0.84089642,
    };
    let radius: i32 = 3;
    let mut total: f32 = 0.0;

    for y in -radius..=radius {
        for x in -radius..=radius {
            let kernel_entry = gb.kernel_at(Vector3::new(x, y, 1));
            print!(" {kernel_entry:.16} |"); //FIXME: is 16 decimal places enough? sure isn't for doubles
            total += kernel_entry;
        }
        println!("");
    }

    println!("Total: {total:.16}");
}
