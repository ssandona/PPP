
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


extern int triangularSmooth(const int width, const int height, const int spectrum, unsigned char * inputImage, unsigned char * smoothImage);


int main(int argc, char *argv[]) {

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

	// Apply the triangular smooth
	CImg< unsigned char > smoothImage = CImg< unsigned char >(inputImage.width(), inputImage.height(), 1, inputImage.spectrum());

	int r=triangularSmooth(inputImage.width(), inputImage.height(), inputImage.spectrum(), inputImage.data(), smoothImage.data());

	if(r==1){
		cout<<"ERR";
		return 1;
	}
	// Save output
	smoothImage.save(("./" + string(argv[1]) + ".smooth2.par.bmp").c_str());

	return 0;
}
