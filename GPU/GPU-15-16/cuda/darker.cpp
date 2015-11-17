
#include <CImg.h>
#include <iostream>
#include <iomanip>
#include <string>

using cimg_library::CImg;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;
using std::string;

extern int darkGray(const int width, const int height, const unsigned char * inputImage, unsigned char * darkGrayImage);


int main(int argc, char *argv[]) {
	int r=0;
	if ( argc != 2 ) {
		cerr << "Usage: " << argv[0] << " <filename>" << endl;
		return 1;
	}

	// Load the input image
	CImg< unsigned char > inputImage = CImg< unsigned char >(argv[1]);
	if ( inputImage.spectrum() != 3 ) {
		cerr << "The input must be a color image." << endl;
		return 1;
	}
	cout << "A";
	// Convert the input image to grayscale and make it darker
	CImg< unsigned char > darkGrayImage = CImg< unsigned char >(inputImage.width(), inputImage.height(), 1, 1);

	r=darkGray(inputImage.width(), inputImage.height(), inputImage.data(), darkGrayImage.data());
	if(r==1){
		cout << "ERRR";
		return 1;
	}
	cout << "Good";
	// Save output
	darkGrayImage.save(("./" + string(argv[1]) + ".dark.seq.bmp").c_str());

	return 0;
}
