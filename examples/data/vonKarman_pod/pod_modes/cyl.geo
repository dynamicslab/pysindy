// Adapted from Martin Einarsve: https://www.youtube.com/watch?v=1A-b84kloFs


// Characteristic lengths
Nx1 = 20; Rx1 = 1.00;  // Upstream top/bottom
Nx2 = 30; Rx2 = 1.0;  // Downstream top/bottom
Ny1 = 20; Ry1 = 1.00;  // Inlet
Ny2 = 10; Ry2 = 1.1;  // Outlet
Nb1  = 20; Rb1  = 1.08;  // Inner radial
Nb2  = 10; Rb2  = 1.05;  // Outer radial (diagonal)
Nb3 = 10;  Rb3 = 1.04;   /// Outer radial (middle)
Nc1  = 20; Rc1  = 1.0;    // Upstream and top/bottom circles
Nc2  = 10; Rc2  = 1.0;   // Downstream circles

// Local densities??
lc1 = 1.0;
lc2 = 1.0;


// Boundary points
scale = 0.5;
Point(1) = {-scale*10, -scale*10, 0, lc2};
Point(2) = {-scale*10, scale*10, 0, lc2};
Point(3) = {scale*30, scale*10, 0, lc2};
Point(4) = {scale*30, 0, 0, lc2};
Point(5) = {scale*30, -scale*10, 0, lc2};
Point(6) = {scale*10, -scale*10, 0, lc2};
Point(7) = {scale*10, 0, 0, lc2};
Point(8) = {scale*10, scale*10, 0, lc2};

// Cylinder boundaries
Point(9) = {0, 0, 0, lc1};
Point(10) = {-0.35355339, -0.35355339, 0, lc1};
Point(11) = {-0.35355339, 0.35355339, 0, lc1};
Point(12) = {0.35355339, 0.35355339, 0, lc1};
Point(13) = {0.5, 0, 0, lc1};
Point(14) = {0.35355339, -0.35355339, 0, lc1};

// Outer circle boundaries
rad = scale*10;
Point(15) = {-rad*0.35355339, -rad*0.35355339, 0, lc1};
Point(16) = {-rad*0.35355339, rad*0.35355339, 0, lc1};
Point(17) = {rad*0.35355339, rad*0.35355339, 0, lc1};
Point(18) = {rad*0.35355339, -rad*0.35355339, 0, lc1};
Point(19) = {rad*0.5, 0, 0, lc1};

//
Line(1) = {1, 6}; Transfinite Curve {1} = Nx1 Using Progression Rx1;
Line(2) = {5, 4}; Transfinite Curve {2} = Ny2 Using Progression 1/Ry2;
Line(3) = {4, 3}; Transfinite Curve {3} = Ny2 Using Progression Ry2;
Line(4) = {2, 8}; Transfinite Curve {4} = Nx1 Using Progression Rx1;
Line(5) = {8, 3}; Transfinite Curve {5} = Nx2 Using Progression Rx2;
Line(6) = {4, 7}; Transfinite Curve {6} = Nx2 Using Progression 1/Rx2;
Line(7) = {5, 6}; Transfinite Curve {7} = Nx2 Using Progression 1/Rx2;
Line(8) = {1, 2}; Transfinite Curve {8} = Ny1 Using Bump Ry1;
Line(9) = {6, 7}; Transfinite Curve {9} = Ny2 Using Progression 1/Ry2;
Line(10) = {7, 8}; Transfinite Curve {10} = Ny2 Using Progression Ry2;


// Cylinder Lines
Circle(11) = {10, 9, 11};  Transfinite Curve {11} = Nc1 Using Progression Rc1;
Circle(12) = {11, 9, 12};  Transfinite Curve {12} = Nc1 Using Progression Rc1;
Circle(13) = {12, 9, 13};  Transfinite Curve {13} = Nc2 Using Progression Rc1;
Circle(14) = {13, 9, 14}; Transfinite Curve {14} = Nc2 Using Progression Rc1;
Circle(15) = {14, 9, 10}; Transfinite Curve {15} = Nc1 Using Progression Rc1;

// Outer block lines
Line(16) = {15, 1};  Transfinite Curve {16} = Nb2 Using Progression Rb2;
Line(17) = {19, 7};  Transfinite Curve {17} = Nb3 Using Progression Rb3;
Line(18) = {18, 6};  Transfinite Curve {18} = Nb2 Using Progression Rb2;  //**
Line(19) = {16, 2};  Transfinite Curve {19} = Nb2 Using Progression Rb2;
Line(20) = {17, 8};  Transfinite Curve {20} = Nb2 Using Progression Rb2;  //**

// Outer circle lines
Circle(21) = {19, 9, 17};  Transfinite Curve {21} = Nc2 Using Progression Rc2;
Circle(22) = {17, 9, 16};  Transfinite Curve {22} = Nc1 Using Progression Rc1;
Circle(23) = {16, 9, 15};  Transfinite Curve {23} = Nc1 Using Progression Rc1;
Circle(24) = {15, 9, 18}; Transfinite Curve {24} = Nc1 Using Progression Rc1;
Circle(25) = {18, 9, 19}; Transfinite Curve {25} = Nc2 Using Progression 1/Rc2;

// Inner block Lines
Line(26) = {10, 15};  Transfinite Curve {26} = Nb1 Using Progression Rb1;
Line(27) = {13, 19};  Transfinite Curve {27} = Nb1 Using Progression Rb1;
Line(28) = {12, 17};  Transfinite Curve {28} = Nb1 Using Progression Rb1;
Line(29) = {11, 16};  Transfinite Curve {29} = Nb1 Using Progression Rb1;
Line(30) = {14, 18};  Transfinite Curve {30} = Nb1 Using Progression Rb1;


// Surfaces


// Outer blocks
Line Loop(1) = {-21, 17, 10, -20};
Plane Surface(1) = {1};
Line Loop(2) = {-24, 16, 1, -18};
Plane Surface(2) = {2};
Line Loop(3) = {-22, 20, -4, -19};
Plane Surface(3) = {3};
Line Loop(4) = {-23, 19, -8, -16};
Plane Surface(4) = {4};
Line Loop(5) = {-25, 18, 9, -17};
Plane Surface(5) = {5};

// Downstream blocks
Line Loop(6) = {-9, -7, 2, 6};
Plane Surface(6) = {6};
Line Loop(7) = {-10, -6, 3, -5};
Plane Surface(7) = {7};

// Inner blocks
Line Loop(8) = {14, 30, 25, -27};
Plane Surface(8) = {8};
Line Loop(9) = {13, 27, 21, -28};
Plane Surface(9) = {9};
Line Loop(10) = {12, 28, 22, -29};
Plane Surface(10) = {10};
Line Loop(11) = {11, 29, 23, -26};
Plane Surface(11) = {11};
Line Loop(12) = {15, 26, 24, -30};
Plane Surface(12) = {12};

Transfinite Surface {1};
Transfinite Surface {2};
Transfinite Surface {3};
Transfinite Surface {4};
Transfinite Surface {5};
Transfinite Surface {6};
Transfinite Surface {7};
Transfinite Surface {8};
Transfinite Surface {9};
Transfinite Surface {10};
Transfinite Surface {11};
Transfinite Surface {12};

Recombine Surface {1};
Recombine Surface {2};
Recombine Surface {3};
Recombine Surface {4};
Recombine Surface {5};
Recombine Surface {6};
Recombine Surface {7};
Recombine Surface {8};
Recombine Surface {9};
Recombine Surface {10};
Recombine Surface {11};
Recombine Surface {12};


Physical Line("Inlet") = {8};
Physical Line("Outlet") = {2, 3};
Physical Line("Wall") = {11, 12, 13, 14, 15};
Physical Line("Top") = {4, 5};
Physical Line("Bottom") = {1, 7};
Physical Surface("Fluid") = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
