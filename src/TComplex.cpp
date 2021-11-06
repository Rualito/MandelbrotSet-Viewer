#include "TComplex.hpp"

TComplex::TComplex(double real, double imag){
    re = real;
    im = imag;
}

TComplex::TComplex(const TComplex& cmp){
    re = cmp.re;
    im = cmp.im;
}

const TComplex& TComplex::operator=(const TComplex& cmp){
    re = cmp.re;
    im = cmp.im;
    return (*this);
}

TComplex TComplex::operator-() const{
    return TComplex(-re, -im);
}
// operator+
TComplex TComplex::operator+(const TComplex& q) const{
    return TComplex(re + q.re, im+q.im);
}

TComplex TComplex::operator+(const double& q)const{
    return TComplex(re + q, im);
}

// operator-
TComplex TComplex::operator-(const TComplex& q)const{
    return TComplex(re-q.re, im-q.im);
}
TComplex TComplex::operator-(const double& q)const{
    return TComplex(re-q, im);
}

// operator*
TComplex TComplex::operator*(const TComplex& q)const{
    return TComplex(re*q.re - im*q.im, im*q.re+re*q.im);
}

TComplex TComplex::operator*(const double& q)const{
    return TComplex(re*q, im*q);
}
// operator/
TComplex TComplex::operator/(const double& q)const{
    	return TComplex(re/q, im/q);
}
TComplex TComplex::operator/(const TComplex& q)const{
    	return TComplex(re*q.re + im*q.im, im*q.re - re*q.im)/(q.re*q.re + q.im*q.im);
}


const TComplex& TComplex::operator+=(const TComplex& q){
	re += q.re;
     	im += q.im;
	return *this;	
} 
const TComplex& TComplex::operator+=(const double& q){
	re += q;
	return *this;	
} 


const TComplex& TComplex::operator-=(const TComplex& q){
	re -= q.re;
     	im -= q.im;
	return *this;	
}
const TComplex& TComplex::operator-=(const double& q){
	re -= q;
	return *this;	
}


const TComplex& TComplex::operator*=(const TComplex& q){
	re = re*q.re-im*q.im;
	im = im*q.re+re*q.im;
	return *this;	
}
const TComplex& TComplex::operator*=(const double& q){
	re *= q;
	im *= q;
	return *this;	
}


const TComplex& TComplex::operator/=(const TComplex& q){
    re = (re*q.re + im*q.im)/(q.re*q.re + q.im*q.im);
    im = (im*q.re - re*q.im)/(q.re*q.re + q.im*q.im);
    return *this;	
}
const TComplex& TComplex::operator/=(const double& q){
    re /= q;
    im /= q;
    return *this;	
}


double TComplex::getArg() const{
    if(re>0)
	return atan(im/re);
    if(re<0){
	if(im>0)
	    return atan(im/re)+M_PI;
        if(im<0)
	    return atan(im/re)-M_PI;
	return M_PI;
    }
    // if re==0
    if(im>0)
	return M_PI/2;
    if(im<0)
	return -M_PI/2;
    //if re=im=0
    // return NaN
    return std::numeric_limits<double>::quiet_NaN();
}

double TComplex::getR() const{
    return sqrt(re*re+im*im);
}
