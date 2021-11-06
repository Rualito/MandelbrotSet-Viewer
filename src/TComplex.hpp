#ifndef TCOMPLEX_H
#define TCOMPLEX_H

#include <cmath>
#include <limits>

using namespace std;

class TComplex{
public:
    TComplex(double real=0, double imag=0);
    
    
    TComplex(const TComplex&);

    const TComplex& operator=(const TComplex&);
    
    TComplex operator+(const TComplex&) const;
    TComplex operator-(const TComplex&) const;
    TComplex operator*(const TComplex&) const;
    TComplex operator/(const TComplex&) const;

    TComplex operator+(const double&) const;
    TComplex operator-(const double&) const;
    TComplex operator*(const double&) const;
    TComplex operator/(const double&) const;

    TComplex operator-() const;
    
    const TComplex& operator+=(const TComplex&);
    const TComplex& operator-=(const TComplex&);
    const TComplex& operator*=(const TComplex&);
    const TComplex& operator/=(const TComplex&);

    const TComplex& operator+=(const double&);
    const TComplex& operator-=(const double&);
    const TComplex& operator*=(const double&);
    const TComplex& operator/=(const double&);

    double getArg() const;
    double getR() const;
    
    double re;
    double im;
};

/*
TComplex operator+(double a, const TComplex& b){
    return (b+a);
}
TComplex operator-(double a, const TComplex& b){
    return TComplex(a-b.re, -b.im);
}
TComplex operator*(double a, const TComplex& b){
    return b*a;
}
TComplex operator/(double a, const TComplex& b){
    return TComplex(a,0)/b;
}
*/
#endif
