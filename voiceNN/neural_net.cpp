#include <stdio.h>
#include <opennn.h>

#include "voicenn.h"
#include "audio_and_data.h"

int main() {

	aad::from_audio_compile_dataset();

	//	Real Data		-->		Sample		--+
	//										  |
	//										  V				  +-->	Discriminator Loss
	//														  |
	//	Random Noise					DISCRIMINATOR		--+
	//		 |							(Deep Convo Net)	--+
	//		 |												  |
	//		 |								  A				  +-->	Generator Loss
	//		 V								  |
	// GENERATOR NN		-->		Sample		--+
	// (Deconvo Net)

	// Discriminator network input
	int d_num_input = aad::get_largest_file_sample();
	int d_num_hidden_neuron;
	int d_num_output;

	OpenNN::NeuralNetwork voice_discriminator(\
		OpenNN::NeuralNetwork::ProjectType::Classification,\
		{d_num_input, d_num_hidden_neuron, d_num_output}\
		);
	voice_discriminator.set_inputs_names();
	voice_discriminator.set_outputs_names();
	OpenNN::ScalingLayer* d_SL_pointer = voice_discriminator.get_scaling_layer_pointer();
	d_SL_pointer->set_descriptives();
	d_SL_pointer->set_scalers();

	// Generative network input
	int g_num_input;
	int g_num_hidden_neuron;
	int g_num_output;

	// Generative network:
	// output: imitation voice (approximation)
	OpenNN::NeuralNetwork voice_generator(\
		OpenNN::NeuralNetwork::ProjectType::Approximation,\
		{g_num_input, g_num_hidden_neuron, g_num_output}\
	);
	voice_generator.set_inputs_names();
	voice_generator.set_outputs_names();

	// data set

	OpenNN::DataSet voice_dataset_real(FILESYSTEM_DATA_PATH, ',', true);
	OpenNN::DataSet voice_dataset_generator_out(FILESYSTEM_GENERATOR_PATH, ',', true);

	// training strategy

	OpenNN::TrainingStrategy voiceNNDiscrimTS(&voice_discriminator, &voice_dataset_real);
	OpenNN::TrainingStrategy voiceNNGenerTS(&voice_generator, &voice_dataset_generator_out);
}