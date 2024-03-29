#include "ColorManager.hpp"

ColorManager::ColorManager(const std::vector<Vector3f>& pnts) {
    for(int i=0;  i<pnts.size()-1; ++i){
	round_P0.push_back(toSpherical(pnts[i]));
	round_C.push_back(toSpherical(pnts[i+1])-round_P0[i]);
    }
}



Vector3f ColorManager::toSpherical(const Vector3f& p) {
    double r = sqrt(p.x*p.x + p.y*p.y + p.z*p.z);

    double phi;
    if(p.x==0)
	phi = M_PI/2;
    else
	phi = atan(p.y/p.x);

    double th;
    if(p.z==0)
        th = M_PI/2;
    else
	th = atan(sqrt(p.x*p.x + p.y*p.y)/p.z);

    return Vector3f(r, th, phi);
}

Vector3f ColorManager::toCartesian(const Vector3f& p) {
    return Vector3f(p.x*sin(p.y)*cos(p.z),
		    p.x*sin(p.y)*sin(p.z),
		    p.x*cos(p.y));
}

void ColorManager::setColors(const std::vector<Vector3f>& pnts){
    round_P0.clear();
    round_C.clear();
    for(int i=0;  i<pnts.size()-1; ++i){
	round_P0.push_back(toSpherical(pnts[i]));
	round_C.push_back(toSpherical(pnts[i+1])-round_P0[i]);
    }

}

Vector3f ColorManager::roundPath(float t)const{
    float segments = round_C.size();

    for(int i=0; i<=segments; ++i){
	if(i/segments>=t){
	    return toCartesian(round_C[i-1]*(t-(i-1)/segments)*segments+round_P0[i-1]);
	}
    }
    // default color
    printf("noooo\n");
    return toCartesian(round_P0[0]);
}
