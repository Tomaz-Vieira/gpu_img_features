use crate::util::BufferSize;

use super::output_buffer::OutputBuffer;

pub struct ReaderBuffer {
    buffer: wgpu::Buffer,
    size: BufferSize,
}
impl ReaderBuffer {
    pub fn new(name: &str, size: BufferSize, device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("read_buffer__{name}")),
            mapped_at_creation: false,
            size: size.padded_size().into(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        });
        return Self { buffer, size };
    }
    pub fn create_for(output_buffer: &OutputBuffer, name: &str, device: &wgpu::Device) -> Self{
        return Self::new(name, output_buffer.size(), device)
    }
    pub fn map_async(
        &self,
        mode: wgpu::MapMode,
        callback: impl FnOnce(Result<(), wgpu::BufferAsyncError>) + wgpu::WasmNotSend + 'static,
    ) {
        let buffer_slice = self.buffer.slice(..); //FIXME
        buffer_slice.map_async(mode, callback);
    }
    pub fn get_mapped_range(&self) -> wgpu::BufferView {
        let buffer_slice = self.buffer.slice(..); //FIXME
        buffer_slice.get_mapped_range()
    }
    pub fn encode_copy(&self, encoder: &mut wgpu::CommandEncoder, source: &OutputBuffer){
        encoder.copy_buffer_to_buffer(source.raw(), 0, &self.buffer, 0, source.size().padded_size().into());
    }
    pub fn unmap(&self){
        self.buffer.unmap()
    }
}

//FIXME: I bet unmap-after-unmap is a panic
// impl Drop for ReaderBuffer {
//     fn drop(&mut self) {
//         self.buffer.unmap()
//     }
// }
