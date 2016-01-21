
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

extern int darkGray(const int width, const int height, const unsigned char * inputImage, unsigned char * darkGrayImage, int pixelThreads);


int main(int argc, char *argv[]) {
	int r=0;
	//cout << "A\n";
	if ( argc != 3 ) {
		//cout << "Err1\n";
		cerr << "Usage: " << argv[0] << " <filename>" << endl;
		return 1;
	}
	//cout << "B\n";
	// Load the input image
	CImg< unsigned char > inputImage = CImg< unsigned char >(argv[1]);
	if ( inputImage.spectrum() != 3 ) {
		cerr << "The input must be a color image." << endl;
		return 1;
	}
	//cout << "C\n";
	// Convert the input image to grayscale and make it darker
	CImg< unsigned char > darkGrayImage = CImg< unsigned char >(inputImage.width(), inputImage.height(), 1, 1);
	//cout << "D\n";
	r=darkGray(inputImage.width(), inputImage.height(), inputImage.data(), darkGrayImage.data(), atoi(argv[2]));
	if(r==1){
		cout << "ERRR\n";
		return 1;
	}
	//cout << "Good\n";
	// Save output
	darkGrayImage.save(("./" + string(argv[1]) + ".dark.uni.more.par.bmp").c_str());

	return 0;
}
