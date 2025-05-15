use std::{marker::PhantomData, time::Instant};

use crate::util::copy_bytes;

/// A buffer that can be mapped to CPU and read back
#[derive(Clone)]
pub struct DownloadBuffer<T>{
    buffer: wgpu::Buffer,
    _marker: PhantomData<T>,
}

pub struct DownloadGuard<T>{
    dl_buffer: DownloadBuffer<T>,
    waiter: flume::Receiver<()>,
    needs_unmapping: bool,
}

impl DownloadBuffer<[f32; 4]>{
    pub fn new_for_predictions(
        input: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
        device: &wgpu::Device,
        label: Option<&str>
    ) -> Self{
        Self::new(
            device,
            label,
            (input.width() * input.height()) as usize,
        )
    }
}

impl<T: bytemuck::AnyBitPattern> DownloadBuffer<T>{
    pub fn new(device: &wgpu::Device, label: Option<&str>, count: usize) -> Self{
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label,
            mapped_at_creation: false,
            size: (count * size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        });
        return Self{buffer, _marker: PhantomData}
    }

    pub fn issue_copy_from(&self, origin: &wgpu::Buffer, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            origin,
            0,
            &self.buffer,
            0,
            origin.size(),
        )
    }

    pub fn map_async(self) -> DownloadGuard<T>{
        let read_buffer_slice = self.buffer.slice(..);
        let start_of_map_async = Instant::now();
        let (tx, rx) = flume::bounded::<()>(1);
        read_buffer_slice.map_async(wgpu::MapMode::Read, move |_| {
            let time_until_mapping = Instant::now() - start_of_map_async;
            eprintln!("Mapping download buffer to CPU memory space took {time_until_mapping:?}");
            tx.send(()).unwrap();
        });
        DownloadGuard { dl_buffer: self,  waiter: rx, needs_unmapping: true}
    }

}

impl<T: bytemuck::AnyBitPattern> DownloadGuard<T>{
    pub fn readback(mut self) -> (Vec<T>, DownloadBuffer<T>){
        self.waiter.recv().unwrap(); //wait for map async to be done
        let out: Vec<T> = {
            let read_buffer_slice = self.dl_buffer.buffer.slice(..);
            let read_buffer_view = read_buffer_slice.get_mapped_range();
            copy_bytes(&read_buffer_view, "from GPU to cpu")
        };
        self.dl_buffer.buffer.unmap();
        self.needs_unmapping = false;
        (out, self.dl_buffer.clone())
    }
}

impl<T> Drop for DownloadGuard<T>{
    fn drop(&mut self) {
        if self.needs_unmapping{ //FIXME: if wgpu kmnows it's not mapped, can't we read that?
            self.dl_buffer.buffer.unmap()
        }
    }
}
