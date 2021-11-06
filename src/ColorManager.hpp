#ifndef COLORMANAGER_H

#define COLORMANAGER_H

//#include <SFML/Vector3.hpp>
#include <SFML/Graphics.hpp>

#include <cmath>
#include <vector>
using namespace sf;

class ColorManager{
public:
    // rgb starting and end points
    ColorManager(const std::vector<Vector3f>&);

    void setColors(const std::vector<Vector3f>&);

    // rgb3 vector
    Vector3f roundPath(float t)const;

    
private:
    std::vector<Vector3f> round_C;
    std::vector<Vector3f> round_P0;
    
    static Vector3f toSpherical(const Vector3f&);
    static Vector3f toCartesian(const Vector3f&);
};

#endif
