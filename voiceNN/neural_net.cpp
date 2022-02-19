#include <stdio.h>
#include <opennn.h>

#include "voicenn.h"
#include "audio_and_data.h"

int main() {

	aad::from_audio_compile_dataset();

	// neural network
	int numInput = aad::get_largest_file_sample();
	int numHiddenNeuron = 0;
	int numTargetVar = 0;

	// Discriminator network:
	// training: real data
	// input: generatorNN output
	// output: predicted voice
	OpenNN::NeuralNetwork voiceNNDiscriminator(\
		OpenNN::NeuralNetwork::ProjectType::Classification,\
		{numInput, numHiddenNeuron, numTargetVar}\
		);

	// Generative network:
	// training:
	// input: noise
	// output: imitation voice (approximation)
	OpenNN::NeuralNetwork voiceNNGenerator(\
		OpenNN::NeuralNetwork::ProjectType::Approximation,\
		{numInput, numHiddenNeuron, numTargetVar}\
		);

	// data set

	OpenNN::DataSet voiceDataSetReal(FILESYSTEM_NOISE_PATH, ',', true);
	OpenNN::DataSet voiceDataSetGenerOut(FILESYSTEM_NOISE_PATH, ',', true);
	// Alternatively, can use these functions to init data set
	// OpenNN::DataSet voiceDataSet;
	// voiceDataSet.set_data_file_name(DATA_SET_FILE_PATH);
	// voiceDataSet.set_separator("Comma");
	// voiceDataSet.load_data();

	// training strategy

	OpenNN::TrainingStrategy voiceNNDiscrimTS(&voiceNNDiscriminator, &voiceDataSetReal);
	OpenNN::TrainingStrategy voiceNNGenerTS(&voiceNNGenerator, &voiceDataSetGenerOut);
}