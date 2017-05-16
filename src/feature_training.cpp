#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int main()
{
  cout << "Reading images..." << endl;
  vector<Mat> images; 
  for(int i=0; i<10; i++)
  {
    string path = "./data"+to_string(i+1)+".png";
  }
}