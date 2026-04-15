import 'package:flutter/material.dart';

class NumberPage extends StatelessWidget {
  const NumberPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0xFF00796B),
        iconTheme: const IconThemeData(
          color: Colors.white, // Set back button color to white
        ),
      ),
      body: Stack(
        children: [
          // Background gradient
          Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  Color(0xFF00796B), // Top gradient color
                  Color(0xFF004D40), // Middle gradient color
                  Colors.black, // Bottom gradient color
                ],
                stops: [0.0, 0.3, 1.0],
              ),
            ),
          ),
          // Content
          Positioned.fill(
            child: Column(
              children: [
                // Header Text inside a gradient box with padding
                Padding(
                  padding: const EdgeInsets.only(left: 30.0, right: 30.0),
                  child: Container(
                    width: double.infinity,
                    decoration: const BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                        colors: [
                          Color(0xFF00796B), // Top color
                          Color(0xFF004D40), // Gradient transition color
                        ],
                        stops: [0.15, 0.67],
                      ),
                      borderRadius: BorderRadius.all(Radius.circular(36.0)),
                    ),
                    padding: const EdgeInsets.symmetric(vertical: 24.0, horizontal: 16.0),
                    child: const Text(
                      "Numbers",
                      style: TextStyle(
                        fontSize: 36, // Increased font size
                        fontWeight: FontWeight.w900, // Extra bold text
                        color: Colors.white,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ),
                ),
                const SizedBox(height: 30),

                // Grid of 10 boxes with numbers 0-9 in a logical layout
                Expanded(
                  child: GridView.builder(
                    padding: const EdgeInsets.all(16.0),
                    gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: 3, // 3 boxes per row
                      crossAxisSpacing: 16.0, // Space between columns
                      mainAxisSpacing: 16.0, // Space between rows
                      childAspectRatio: 1, // Make the boxes square-shaped
                    ),
                    itemCount: 10, // Now we have 10 boxes (0 to 9)
                    itemBuilder: (context, index) {
                      // Define the images for each number box
                      String imagePath = '';
                      if (index == 0) {
                        imagePath = 'assests/zeroSign.png'; // Image for 0
                      } else if (index == 1) {
                        imagePath = 'assests/firstSign.png'; // Image for 1
                      } else if (index == 2) {
                        imagePath = 'assests/twoSign.png'; // Image for 2
                      } else if (index == 3) {
                        imagePath = 'assests/threeSign.png'; // Image for 3
                      } else if (index == 4) {
                        imagePath = 'assests/fourSign.png'; // Image for 4
                      } else if (index == 5) {
                        imagePath = 'assests/fiveSign.png'; // Image for 5
                      } else if (index == 6) {
                        imagePath = 'assests/sixSign.png'; // Image for 6
                      } else if (index == 7) {
                        imagePath = 'assests/sevenSign.png'; // Image for 7
                      } else if (index == 8) {
                        imagePath = 'assests/eightSign.png'; // Image for 8
                      } else if (index == 9) {
                        imagePath = 'assests/nineSign.png'; // Image for 9
                      }

                      return Container(
                        decoration: BoxDecoration(
                          color: Colors.grey.withOpacity(0.3), // Transparent gray for all boxes
                          borderRadius: BorderRadius.circular(16.0),
                        ),
                        child: Stack(
                          children: [
                            // Number in the upper-left corner
                            Positioned(
                              top: 8.0,
                              left: 8.0,
                              child: Text(
                                '$index', // Display the number (0 to 9)
                                style: const TextStyle(
                                  fontSize: 24,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white,
                                ),
                              ),
                            ),
                            // Add the image to the corresponding box
                            if (imagePath.isNotEmpty)
                              Positioned(
                                bottom: 8.0,
                                right: 8.0,
                                left: 12.0, // Adjust width within the box
                                top: 50.0, // Provide top margin for better scaling
                                child: Image.asset(
                                  imagePath, // Correct path for the image
                                  fit: BoxFit.contain, // Ensure the image fits well
                                ),
                              ),
                          ],
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
