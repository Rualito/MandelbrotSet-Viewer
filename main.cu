#include <SFML/Graphics.hpp>

#include "TComplex.hpp"
#include "ColorManager.hpp"

#include <cstdio>
#include <string>

#include "Julia.cuh"
#include "ColorData.cuh"

#define RECURSION_DEPTH 5000
//#define COARSE

#define COUNT
#include "Debug.hpp"

TComplex func(const TComplex& z, const TComplex& c){
    return (c + z*z);
}

// z is the starting point, c is constant
// cpu implementation
double julia_cpu(const TComplex& z, const TComplex& c){

    TComplex temp(z);
    for(int i=0; i<RECURSION_DEPTH; ++i){
	temp = func(temp, c);
	
	if(temp.getR()>3){
	    if(c.re<0 && c.re>-0.1&& fabs(c.im)<0.1){
		// hmmmm
		printf("%f + i %f -> %f\n", c.re, c.im, temp.getR());
	    }
	    return (0.+i)/RECURSION_DEPTH;
	}
    }
    return 1.;
}


double mandelbrot(const TComplex& c){
    return julia_cpu(TComplex(0), c);
}

float colorFunc(float t){
    float d = RECURSION_DEPTH;
    return log(d*t+1)/log(d+1);
}

sf::VertexArray getJulia(const TComplex& z0_, int windowSize[2], double recenter[2], double zoom, const ColorManager& cMandel, int DEPTH){
    // recenter between -1,1
    const int gridN = 1024;
    static double center[2]{0,0};
    
    double rectStep[2]{(0.+windowSize[0])/gridN, (0.+windowSize[1])/gridN}; 

    double ratio = (windowSize[1]+0.)/windowSize[0];
    
    double stdrange = 3.5;
    double xdelta = stdrange/(2*zoom);
    double ydelta = ratio*xdelta;

    center[0]+=2*recenter[0]*xdelta;
    center[1]+=2*recenter[1]*ydelta;
    
    double xrange[2]{center[0] - xdelta,
	             center[0] + xdelta};
    double yrange[2]{center[1] - ydelta,
	             center[1] + ydelta};
    
    sf::VertexArray quads(sf::Quads, 4*(gridN-1)*(gridN-1));

    double z0[2]{z0_.re, z0_.im};

    double c0[2]{xrange[0], yrange[0]};
    double cstep[2]{(xrange[1]-xrange[0])/gridN, (yrange[1]-yrange[0])/gridN};

    float **colorResult = new float*[gridN];
    for(int i=0; i<gridN; ++i){
        colorResult[i] = new float[gridN];
    }
    
    getColorData(gridN, z0, c0, cstep, colorResult, floor(DEPTH));
    
    for(int j=0; j<gridN-1; ++j){
        for(int i=0; i<gridN-1; ++i){
            
            // vertex array of quads
            double px0 = i*rectStep[0];
            double px1 = (i+1)*rectStep[0];
        
            double py0 = j*rectStep[1];
            double py1 = (j+1)*rectStep[1];

            // sets the positions of the corners of the quadrilaterals on the screen
            quads[j*(gridN-1)*4+i*4+0].position = sf::Vector2f(px0, py0);
            quads[j*(gridN-1)*4+i*4+1].position = sf::Vector2f(px1, py0);
            quads[j*(gridN-1)*4+i*4+2].position = sf::Vector2f(px1, py1);
            quads[j*(gridN-1)*4+i*4+3].position = sf::Vector2f(px0, py1);

            Vector3f cId0 = 255.f*cMandel.roundPath(colorFunc(colorResult[i][j]));
            Vector3f cId1 = 255.f*cMandel.roundPath(colorFunc(colorResult[i+1][j]));
            Vector3f cId2 = 255.f*cMandel.roundPath(colorFunc(colorResult[i+1][j+1]));
            Vector3f cId3 = 255.f*cMandel.roundPath(colorFunc(colorResult[i][j+1]));
            if(cId0.y<0){
                printf("crap\n");
            }
            // sets the color of the corners of the quadrilateral
            quads[j*(gridN-1)*4+i*4+0].color = sf::Color(cId0.x, cId0.y, cId0.z);
            quads[j*(gridN-1)*4+i*4+1].color = sf::Color(cId1.x, cId1.y, cId1.z);
            quads[j*(gridN-1)*4+i*4+2].color = sf::Color(cId2.x, cId2.y, cId2.z);
            quads[j*(gridN-1)*4+i*4+3].color = sf::Color(cId3.x, cId3.y, cId3.z);
        }
    }

    for(int i=0; i<gridN; ++i){
    	delete[] colorResult[i];
    }
    delete[] colorResult;
    return quads;
}

double testComplex(const TComplex& c){
    return (c.getArg()/(2*M_PI)+0.5);
}

int main(){
    int windowSize[2]{1200,800};

    
    Vector3f color0(91, 88, 245); color0/=255.f;
    Vector3f color1(121, 232, 230); color1/=255.f;
    Vector3f color2(121, 232, 121); color2/=255.f;
    Vector3f color3(228, 232, 121); color3/=255.f;
    Vector3f color4(245, 149, 120); color4/=255.f;

    std::vector<Vector3f> colorPalette1;
    colorPalette1.push_back(color4);
    colorPalette1.push_back(color3);
    colorPalette1.push_back(color2);
    colorPalette1.push_back(color1);
    colorPalette1.push_back(color0);

    Vector3f color5(0,   7, 100); color5/=255.f;
    Vector3f color6(32, 107, 203); color6/=255.f;
    Vector3f color7(237, 255, 255); color7/=255.f;
    Vector3f color8(255, 170,   0); color8/=255.f;
    Vector3f color9(0,   2,   0); color9/=255.f;

    std::vector<Vector3f> colorPalette2;
    colorPalette2.push_back(color9);
    colorPalette2.push_back(color8);
    colorPalette2.push_back(color7);
    colorPalette2.push_back(color6);
    colorPalette2.push_back(color5);

    ColorManager cMandel(colorPalette2);

    double center[2]{0,0};
    double zoom = 1;
    TComplex julia_c(0, 0.); 
    sf::VertexArray vex = getJulia(julia_c, windowSize, center, zoom, cMandel, RECURSION_DEPTH);
    
    sf::RenderWindow window(sf::VideoMode(windowSize[0],windowSize[1]), "Hello world");
    //sf::CircleShape shape(100.f);
    //shape.setFillColor(sf::Color::Green);

    sf::Font mFont;

    if(!mFont.loadFromFile("Arial.ttf")){
        printf("Couldn't load font..\n");
        exit(-1);
    }
    
    sf::Text text;
    text.setFont(mFont);
    //	("hello", mFont, 18);
    text.setString("Greetings ma dudes\n");
    text.setCharacterSize(14);
    text.setFillColor(sf::Color::Black);
    
    bool mousePressed = false;
    bool addPressed = false;
    bool subPressed = false;
    
    double depth = RECURSION_DEPTH;
    while(window.isOpen()){
        sf::Event event;
        while(window.pollEvent(event)){
            if(event.type == sf::Event::Closed) window.close();
            
        }

        Vector2i mpos = sf::Mouse::getPosition(window);
        
        text.setString("x: "+to_string(mpos.x)+"; y: "+to_string(mpos.y));
        text.setPosition(10.,10.);
        center[0]=0;
        center[1]=0;
        if(sf::Mouse::isButtonPressed(sf::Mouse::Left) && !mousePressed){
            mousePressed = true;
            center[0] = 2*(0.+mpos.x)/windowSize[0]-1;
            center[1] = 2*(0.+mpos.y)/windowSize[1]-1;

            zoom*=3;
            printf("Zoom now at: %f\n", zoom);
            vex = getJulia(julia_c, windowSize, center, zoom, cMandel, floor(depth));
            // zoom in
        } else if(sf::Mouse::isButtonPressed(sf::Mouse::Right) && !mousePressed){
            mousePressed = true;
            zoom/=3;
            center[0] = 2*(0.+mpos.x)/windowSize[0]-1;
            center[1] = 2*(0.+mpos.y)/windowSize[1]-1;

            vex = getJulia(julia_c, windowSize, center, zoom, cMandel, floor(depth));

            //zoom out
        } else if(mousePressed){
            mousePressed = false;
        }

        // if (sf::Keyboard::isKeyPressed(sf::Keyboard::Add) && !addPressed){
        //     addPressed = true;
        //     depth*=1.2;
        //     printf("Recursion now at: %f\n", depth);
        //     vex = getJulia(TComplex(0), windowSize, center, zoom, cMandel, floor(depth));
        // } else if(addPressed){
        //     addPressed=false;
        // }

        // if (sf::Keyboard::isKeyPressed(sf::Keyboard::Subtract) && !subPressed){
        //     subPressed = true;
        //     depth/=1.2;
        //     vex = getJulia(TComplex(0), windowSize, center, zoom, cMandel, floor(depth));
        // } else if(addPressed){
        //     subPressed=false;
        // }

        
        window.clear();
        window.draw(vex);
        window.draw(text);
        window.display();
    }
    //*/
    return 0;
}
