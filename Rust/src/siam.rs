extern crate tensorflow;

use tensorflow::Tensor;

pub struct Siamese {
	seq_length: u32,
	batch_size: u32,
	layers: u32
}

impl Siamese {
	pub fn init() {
		
	}

	pub fn rnn(input_x: Tensor<f32>, name: String){

	}
}
