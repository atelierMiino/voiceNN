#include <fstream>
#include <filesystem>
#include <string>
#include <io.hpp>

#include "voicenn.h"

namespace aad {
	unsigned int largest_file_sample = 0;

	// generates data set based on MP3 files
	void from_audio_compile_dataset() {

		std::string prefix = "../MP3/";
		std::string suffix = ".mp3";

		std::ofstream output_file;
		output_file.open(FILESYSTEM_DATA_PATH);

		// read all mp3 files in a folder, one by one
		for (const auto& entry : std::filesystem::directory_iterator(FILESYSTEM_MP3_PATH)) {
			std::string entry_string = entry.path().u8string();
			kfr::audio_reader_mp3<float> reader(kfr::open_file_for_reading(entry_string));
			kfr::univector2d<float> vaudio = reader.read_channels();

			// remove path from filename and list as first entry in each row in CSV file
			std::string name = entry_string;
			name.erase(0, prefix.length());
			int suffix_beginning = name.find(suffix);
			name.erase(suffix_beginning, suffix.length());
			output_file << name;

			// write each entry into the CSV file
			int i = 0;
			for (; i < vaudio[0].size(); i++) {
				output_file << "," << vaudio[0].at(i);
			}

			// record largest file sample
			if (i > largest_file_sample) { largest_file_sample = i; }

			// new line for every audio entry
			output_file << "\n";
		}

		output_file.close();
	}

	// function is used to make sure the NN has enough space to work with
	// all other smaller files will be 0-padded inputs
	int get_largest_file_sample() {
		return largest_file_sample;
	}
}