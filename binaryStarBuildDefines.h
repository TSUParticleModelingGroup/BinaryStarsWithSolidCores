//How big the blocks will be.
#define BLOCKSIZE 128

//A good number for pi.
#define PI 3.141592654f

//If number is smaller than this assume it is zero.
#define ASSUME_ZERO_FLOAT 0.000001f
#define ASSUME_ZERO_DOUBLE 0.000000000001
		
//Number of outer elements to us when finding radius
#define NUMBER_ELEMENTS_RADIUS 100	

//Universal gravitational constant in kilometersE3*kilogramsE-1*secondsE-2
#define UNIVERSAL_GRAVITY_CONSTANT 6.67408e-20

//Radius of the Sun in kilometers
#define DIAMETER_SUN 1391020

//The total mass of the Sun in kilograms
#define MASS_SUN 1.989e30

//How many times bigger the main sequence star can grow as it becomes a red giant.
#define RED_GIANT_GROWTH 50

//Repultion strengths of the plasma of star1 in kilograms*kilometersE-1*secondsE-2. WIll turn an area into a force.
#define PUSH_BACK_PLASMA1 2.0e14

//Repultion strengths of the plasma of star2 in kilograms*kilometersE-1*secondsE-2. WIll turn an area into a force.
#define PUSH_BACK_PLASMA2 2.0e14

//This will be multiplier of the force of gravity when a plasma element first touches a core element.
#define PUSH_BACK_CORE_MULT1 1.0f

//This will be multiplier of the force of gravity when a plasma element first touches a core element.
#define PUSH_BACK_CORE_MULT2 1.0f

//How much to reduce the push back when plasma-plasma elements are retreating (it is multiplied by the push back strenght).
#define PLASMA_PUSH_BACK_REDUCTION 0.1

//How much to reduce the push back when plasma-core elements are retreating (it is multiplied by the push back strenght).
#define CORE_PUSH_BACK_REDUCTION 0.9	

//Radius tolerance in percent off desired value.
#define DIAMETER_TOLERANCE 0.2

//How fast you try to adjust the push back to reach the tolerance of the diameter of star1.
#define PUSH_BACK_ADJUSTMENT1 5.0

//How fast you try to adjust the push back to reach the tolerance of the diameter of star2.
#define PUSH_BACK_ADJUSTMENT2 5.0

//The maximum randium speed given to the intially created elements to help remove any bias in the stars create because 
//they were generated on a cube. Speed in kilometers per second.
#define MAX_INITIAL_PLASMA_SPEED 50.0

//Start damping amount used in settling star cubes into spheres bodies:
#define DAMP_AMOUNT 50.0

//Time to damp the raw stars. In days (24 hour period): This will be done NumberOfDampIncriments times.
#define DAMP_TIME 1.0

//Number of iterations to decrease the damp amount to zero.
#define DAMP_INCRIMENTS 10

//Time to let the damping settle down. In days (24 hour period).
#define DAMP_REST_TIME 1.0

//Time to adjust the radius in days (24 hour period).
#define RADIUS_ADJUSTMENT_TIME 1.0

//Time to let the radius adjust settle out in days (24 hour period).
#define RADIUS_ADJUSTMENT_REST_TIME 1.0

//Time to let the spins settle down. In days (24 hour period).
#define SPIN_REST_TIME 2.0

//Time step as a fraction of a time unit.
#define DT 0.002 			