#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

#include <boost/timer.hpp>

#include <sophus/se3.h>
using Sophus::SE3;

#include <Eigen/Dense>
// #include <eigen3/Eigen/Core>
// #include <eigen3/Eigen/Geometry>
using namespace Eigen;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

const int border = 20; 	// 边缘宽度
const int width = 640;  	// 宽度 
const int height = 480;  	// 高度
const double fx = 481.2f;	// 相机内参
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 2;	// NCC 取的窗口半宽度
const int ncc_area = (2*ncc_window_size+1)*(2*ncc_window_size+1); // NCC窗口面积
const double min_cov = 0.1;	// 收敛判定：最小方差
const double max_cov = 10;	// 发散判定：最大方差

// 像素点转换到相机坐标系下归一化平面空间点，同时乘以深度，即可得到空间点坐标
inline Vector3d pixel2cam(const Vector2d& px)
{
  return Vector3d( (px(0,0)-cx)/fx, 
		   (px(1,0)-cy)/fy, 
		   1
		 );
}

// 相机坐标系下空间点转换到像素点
inline Vector2d cam2pixel(const Vector3d& p_cam)
{
  return Vector2d( p_cam(0,0)/p_cam(2,0)*fx+cx,
		   p_cam(1,0)/p_cam(2,0)*fy+cy);
}

inline bool inside(const Vector2d& ptr)
{
  return (ptr(0,0)<=width-border && ptr(0,0)>=border 
	  && ptr(0,1)<=height-border && ptr(0,1)>=border
	 );
}


bool readDatasetFile(const string& path_to_dataset, 
		     vector<string>& vColor_image_filenames, 
		     vector<SE3>& vPoses_Twc 
		    );

bool update(const Mat& ref, 
	    const Mat& cur,
	    const SE3 Trc,
	    Mat& depth,
	    Mat& depth_cov2
	   );

inline bool plotDepthImage(const Mat& depth)
{
  imshow("depth", depth*0.4);
  waitKey(1);
  return true;
}

bool epipolarSearch(const Mat& ref, 
		    const Mat& cur, 
		    const SE3& Tcr,
		    const double& depth,
		    const double& depth_cov2,
		    const Vector2d& ptr_ref, 
		    Vector2d& ptr_cur
		   );

double ZNCC(const Mat& ref, 
	    const Mat& cur, 
	    const Vector2d& ptr_ref, 
	    const Vector2d& ptr_cur
	   );


double getBilinearInterpolatedValue( const Mat& img, const Vector2d& pt );


bool updateDepthFilter(const Vector2d& ptr_ref, 
		       const Vector2d& ptr_cur,
		       const SE3& Tcr, 
		       Mat& depth, 
		       Mat& depth_cov2
		      );

int main(int argc, char** argv)
{
  if(argc!=2)
  {
    cerr << "Use: point_cloud path_to_dataset" << endl;
    return -1;
  }
  
  vector<string> vColor_image_filenames;
  vector<SE3> vPoses_Twc;
  bool ret = readDatasetFile(argv[1], vColor_image_filenames, vPoses_Twc);
  if(!ret)
  {
    cerr << "Cannot load images..." << endl;
    return -1;
  }
  cout << "Total " << vColor_image_filenames.size() << " images. (should be 200)" << endl;
  
  
  Mat ref = imread(vColor_image_filenames[0],0);// grayscale
  SE3 ref_Twc = vPoses_Twc[0];
  double init_depth = 3.0;
  double init_cov2 = 3.0;
  
  Mat depth(height, width, CV_64F, init_depth);
  Mat depth_cov2(height, width, CV_64F, init_cov2);
  
  for(int i=1; i<vColor_image_filenames.size(); i++)
  {
    cout << "Depth filtering in image " << i << "..." << endl;
    Mat cur = imread(vColor_image_filenames[i],0);
    if(cur.data == nullptr) 
      continue;
    SE3 cur_Twc = vPoses_Twc[i];
    SE3 Tcr = cur_Twc.inverse()*ref_Twc; // tranform from ref to cur
    update(ref, cur, Tcr, depth, depth_cov2);
    plotDepthImage(depth);
    imshow("current image", cur);
    waitKey(0);
  }
  
  cout << "Finishing depth filtering, saving to file depth.png..." << endl;
  imwrite("depth.png", depth);
  cout << "Done." << endl;
//   imshow("init_depth",depth);
//   waitKey(0);
  return 0;
}

bool readDatasetFile(const string& path_to_dataset, 
		     vector<string>& vColor_image_filenames, 
		     vector<SE3>& vPoses_Twc 
		    )
{
  ifstream fin(path_to_dataset+"first_200_frames_traj_over_table_input_sequence.txt");
  if(!fin) 
    return false;
  
  while(!fin.eof())
  {
    string filename;
    fin>>filename;
    vColor_image_filenames.push_back(path_to_dataset+"/images/"+filename);
    
    double data[7];
    for(double& d:data) 
      fin>>d;
    vPoses_Twc.push_back(
      SE3(Quaterniond(data[6],data[3],data[4],data[5]),
	  Vector3d(data[0],data[1],data[2])
      )
    );
    
    if(!fin.good())
      break;
  }
  return true;
}


// depth filter
bool update(const Mat& ref, const Mat& cur, const SE3 Tcr, Mat& depth, Mat& depth_cov2)
{
  for(int u = border; u<width-border; u++)
    for(int v = border; v<height-border; v++)
    {
      if(depth_cov2.ptr<double>(v)[u] < min_cov || depth_cov2.ptr<double>(v)[u] > max_cov)
	continue;
      Vector2d ptr_cur;
      bool ret = epipolarSearch(ref, 
				cur, 
				Tcr, 
				depth.ptr<double>(v)[u], 
				depth_cov2.ptr<double>(v)[u], 
				Vector2d(u,v), 
				ptr_cur);
      if(!ret) continue;
      updateDepthFilter(Vector2d(u,v), ptr_cur, Tcr, depth, depth_cov2);
    }    
}

bool epipolarSearch(const Mat& ref, 
		    const Mat& cur, 
		    const SE3& Tcr, 
		    double& depth, 
		    double& depth_cov2,
		    const Vector2d& ptr_ref, 
		    Vector2d& ptr_cur)
{
  Vector3d p_cam = pixel2cam(ptr_ref);
  p_cam.normalize();// inplace normalization
  Vector3d p = p_cam*depth;
  
  double d_min = depth - 3*depth_cov2; // 3倍均方差
  double d_max = depth + 3*depth_cov2;
  if(d_min<0.1) 
    d_min =0.1;
  Vector2d ptr_cur_mean = cam2pixel(Tcr*p);
  Vector2d ptr_cur_min = cam2pixel(Tcr*(p_cam*d_min));
  Vector2d ptr_cur_max = cam2pixel(Tcr*(p_cam*d_max));
  // 极线
  Vector2d epipolar_line = ptr_cur_max - ptr_cur_min;
  double epipolar_line_length = epipolar_line.norm();
  Vector2d epipolar_line_direction = epipolar_line.normalized();
  if(epipolar_line_length > 200) 
    epipolar_line_length = 200;
  
  // NCC patch matching
  double best_ncc = -0.1;
  Vector2d ptr_best_match;
  for(double l=-epipolar_line_length/2; l<= epipolar_line_length/2; l+=0.7)
  {
    Vector2d ptr_moving = ptr_cur_mean + l*epipolar_line_direction;
    if(!inside(ptr_moving))
      continue;
    double ncc = ZNCC(ref, cur, ptr_ref, ptr_moving);
    if(ncc>best_ncc)
    {
      best_ncc = ncc;
      ptr_best_match = ptr_moving;
    }
    if(best_ncc<0.85f)
      return false;
    ptr_cur = ptr_best_match;
    return true;
  }
}


// const int ncc_window_size = 2;	// NCC 取的窗口半宽度
// const int ncc_area = (2*ncc_window_size+1)*(2*ncc_window_size+1); // NCC窗口面积
double ZNCC(const Mat& ref, const Mat& cur, const Vector2d& ptr_ref, const Vector2d& ptr_cur)
{
  double ref_mean=0, cur_mean=0;
  vector<double> ref_pixelValues, cur_pixelValues;
  for(int i = -ncc_window_size; i<=ncc_window_size; i++)
    for(int j = -ncc_window_size; j<=ncc_window_size; j++)
    {
      double ref_pixelValue = double( ref.ptr<uchar>(int(ptr_ref(1,0)+j))[int(ptr_ref(0,0)+i)] )/255.0;
      ref_pixelValues.push_back(ref_pixelValue);
      ref_mean += ref_pixelValue;
      
      double cur_pixelValue = getBilinearInterpolatedValue( cur, ptr_cur+Vector2d(i,j) );
      cur_pixelValues.push_back(cur_pixelValue);
      cur_mean += cur_pixelValue;
    }
  ref_mean /= ncc_area;
  cur_mean /= ncc_area;
  
  double numerator, denominator1, denominator2;
  for(int i=0; i<ncc_area; i++)
  {
    numerator += (ref_pixelValues[i] - ref_mean)*(cur_pixelValues[i] - cur_mean);
    denominator1 += (ref_pixelValues[i] - ref_mean)*(ref_pixelValues[i] - ref_mean);
    denominator2 += (cur_pixelValues[i] - cur_mean)*(cur_pixelValues[i] - cur_mean);
    
    double denomenator = sqrt(denominator1*denominator2);
    if(denomenator<1e-10)
      return -1;//TODO
    else
      return numerator/denomenator;
  }
  
}

double getBilinearInterpolatedValue( const Mat& img, const Vector2d& pt )
{
  uchar* data = &img.data[int(pt(1,0))*img.step+int(pt(0,0))];
  double xx = pt(0,0) - floor(pt(0,0));
  double yy = pt(0,1) - floor(pt(0,1));
  return (1-xx)*(1-yy)*data[0] + xx*(1-yy)*data[1] + (1-xx)*yy*data[img.step] + xx*yy*data[img.step+1];
}

bool updateDepthFilter(
    const Vector2d& pt_ref, 
    const Vector2d& pt_curr, 
    const SE3& T_C_R,
    Mat& depth, 
    Mat& depth_cov
)
{
    // 我是一只喵
    // 不知道这段还有没有人看
    // 用三角化计算深度
    SE3 T_R_C = T_C_R.inverse();
    Vector3d f_ref = pixel2cam( pt_ref );
    f_ref.normalize();
    Vector3d f_curr = pixel2cam( pt_curr );
    f_curr.normalize();
    
	// 方程参照本书第 7 讲三角化一节
    Vector3d t = T_R_C.translation();
    Vector3d f2 = T_R_C.rotation_matrix() * f_curr; 
    Vector2d b = Vector2d ( t.dot ( f_ref ), t.dot ( f2 ) );
    double A[4];
    A[0] = f_ref.dot ( f_ref );
    A[2] = f_ref.dot ( f2 );
    A[1] = -A[2];
    A[3] = - f2.dot ( f2 );
    double d = A[0]*A[3]-A[1]*A[2];
    Vector2d lambdavec = 
        Vector2d (  A[3] * b ( 0,0 ) - A[1] * b ( 1,0 ),
                    -A[2] * b ( 0,0 ) + A[0] * b ( 1,0 )) /d;
    Vector3d xm = lambdavec ( 0,0 ) * f_ref;
    Vector3d xn = t + lambdavec ( 1,0 ) * f2;
    Vector3d d_esti = ( xm+xn ) / 2.0;  // 三角化算得的深度向量
    double depth_estimation = d_esti.norm();   // 深度值
    
    // 计算不确定性（以一个像素为误差）
    Vector3d p = f_ref*depth_estimation;
    Vector3d a = p - t; 
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos( f_ref.dot(t)/t_norm );
    double beta = acos( -a.dot(t)/(a_norm*t_norm));
    double beta_prime = beta + atan(1/fx);
    double gamma = M_PI - alpha - beta_prime;
    double p_prime = t_norm * sin(beta_prime) / sin(gamma);
    double d_cov = p_prime - depth_estimation; 
    double d_cov2 = d_cov*d_cov;
    
    // 高斯融合
    double mu = depth.ptr<double>( int(pt_ref(1,0)) )[ int(pt_ref(0,0)) ];
    double sigma2 = depth_cov.ptr<double>( int(pt_ref(1,0)) )[ int(pt_ref(0,0)) ];
    
    double mu_fuse = (d_cov2*mu+sigma2*depth_estimation) / ( sigma2+d_cov2);
    double sigma_fuse2 = ( sigma2 * d_cov2 ) / ( sigma2 + d_cov2 );
    
    depth.ptr<double>( int(pt_ref(1,0)) )[ int(pt_ref(0,0)) ] = mu_fuse; 
    depth_cov.ptr<double>( int(pt_ref(1,0)) )[ int(pt_ref(0,0)) ] = sigma_fuse2;
    
    return true;
}





