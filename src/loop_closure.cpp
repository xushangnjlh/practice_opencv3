#include <DBoW3/DBoW3.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <iostream>
using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    cout << "Reading database..." << endl;
    DBoW3::Vocabulary vocab("vocab_larger_my.yml.gz");
    if(vocab.empty())
    {
        cerr << "Cannot load vacabulary!" << endl;
        return -1;
    }

    cout << "Reading images..." << endl;
    vector<Mat> images;
    for(int i=0; i<10; i++)
    {
        string path = "../data/" + to_string(i+1) + ".png";
        images.push_back(imread(path));
    }

    cout << "Detecting features..." << endl;
    Ptr<Feature2D> detector = ORB::create();
    vector<Mat> descriptors;
    for(int i=0; i<10; i++)
    {
        vector<KeyPoint> keyPoints;
        Mat descriptor;
        detector->detectAndCompute(images[i], Mat(), keyPoints, descriptor);
        descriptors.push_back(descriptor);
    }

    cout << "Compare image with image..." << endl;
    for(int i=0; i<images.size(); i++)
    {
        DBoW3::BowVector v1;
        vocab.transform(descriptors[i],v1);
        for(int j=i; j<images.size(); j++)
        {
            DBoW3::BowVector v2;
            vocab.transform(descriptors[j], v2);
            double score = vocab.score(v1,v2);
            cout << "image " << i << " vs image " << j << " : " << score << endl;
        }
        cout << endl;
    }

    cout << "Compare image with database..." << endl;
    DBoW3::Database db(vocab, false, 0);
    cout << "Database info: " << db << endl;
    for(int i=0; i<descriptors.size(); i++)
    {
        db.add(descriptors[i]);
    }
    cout << "Database info: " << db << endl;
    for(int i=0; i<descriptors.size(); i++)
    {
        DBoW3::QueryResults ret;
        db.query(descriptors[i],ret, 4);
        cout << "Searching for image " << i << " returns " << ret << endl << endl;
    }
    cout << "Done." << endl;
    return 0;
}
















