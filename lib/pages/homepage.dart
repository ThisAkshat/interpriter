import 'package:flutter/material.dart';
import '../bloc.navigation_bloc/navigation_bloc.dart';
import 'offline_mode.dart';
import 'chatSupport.dart'; // Import your chatSupport.dart file
import 'ATS.dart'; // Import your ATS.dart file (make sure ATS.dart exists)
import 'STT.dart'; // Import your STT.dart file (make sure STT.dart exists)
import 'ABCI.dart'; // Import the ABCI.dart file
import 'more.dart'; // Import the new more.dart file
import 'sign_detector_page.dart';

class HomePage extends StatelessWidget implements NavigationStates {
  const HomePage({super.key});


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFBFC4C4),
      body: Stack(
        children: [
          // Split background colors
          Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  Color(0xFF00796B), // Bottom color now at the top
                  Color(0xFF004D40), // Gradient transition color
                  Colors.black,      // Black color now at the bottom
                ],
                stops: [0.0, 0.3, 1.0], // Adjust stops as needed
              ),
            ),
          ),

          // Main content
          Positioned.fill(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: 40), // For top padding

                // Search Box
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const SizedBox(width: 0),
                    Container(
                      height: 40,
                      width: 300,
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(20), // Circular from both sides
                      ),
                      child: const Row(
                        crossAxisAlignment: CrossAxisAlignment.center,
                        children: [
                          Icon(Icons.search, color: Colors.grey),
                          SizedBox(width: 8),
                          Expanded(
                            child: TextField(
                              decoration: InputDecoration(
                                hintText: 'Search',
                                hintStyle: TextStyle(height: 1.8),
                                border: InputBorder.none,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),

                const SizedBox(height: 60), // Adjust this value for the desired distance

                // Sign Subtitle Box
                Center(
                  child: Container(
                    padding: const EdgeInsets.all(40),
                    margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 20),
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                        colors: [
                          Color(0xFF00796B), // Top color
                          Color(0xFF004D40), // Bottom color
                        ],
                        stops: [0.15, 0.85],
                      ),
                      borderRadius: BorderRadius.circular(40),
                    ),
                    child: const Column(
                      children: [
                        Text(
                          'Sign subtitle',
                          style: TextStyle(
                            fontSize: 30,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                        SizedBox(height: 12),
                        Text(
                          'Try our special features with the sign subtitles',
                          style: TextStyle(
                            fontSize: 18,
                            color: Colors.white,
                          ),
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 30),

                // Centered Circles
                Expanded(
                  child: Center(
                    child: Padding(
                      padding: const EdgeInsets.only(top: 15),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          _buildCircle(
                            icon: Icons.pan_tool,
                            label: 'Sign to Text',
                            onTap: () {
                              // Navigate to STT page (Sign to Text)
                              Navigator.push(
                                context,
                                MaterialPageRoute(builder: (context) => const STTPage()),
                              );
                            },
                          ),
                          const SizedBox(width: 16),
                          _buildCircle(
                              icon: Icons.mic,
                              label: 'Audio to Sign',
                              onTap: () {
                                // Navigate to ATS page (Audio to Sign)
                                Navigator.push(
                                  context,
                                  MaterialPageRoute(builder: (context) => const ATSPage()),
                                );
                              }
                          ),
                          const SizedBox(width: 16),
                          _buildCircle(
                            icon: Icons.widgets_sharp,
                            label: 'ABCI',
                            onTap: () {
                              // Navigate to ABCIPage
                              Navigator.push(
                                context,
                                MaterialPageRoute(builder: (context) => const ABCIPage()),
                              );
                            },
                          ),
                          const SizedBox(width: 16),
                          _buildCircle(
                            icon: Icons.more_horiz,
                            label: 'MORE',
                            onTap: () {
                              // Navigate to MorePage
                              Navigator.push(
                                context,
                                MaterialPageRoute(builder: (context) => const MorePage()),
                              );
                            },
                          ),
                        ],
                      ),
                    ),
                  ),
                ),

                // Advertisement Box
                const SizedBox(height: 10),
                Container(
                  height: 220,
                  margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  padding: const EdgeInsets.only(bottom: 16),
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                      colors: [
                        Color(0xFF00796B), // Top color
                        Color(0xFF004D40), // Bottom color
                      ],
                      stops: [0.15, 0.85],
                    ),
                    borderRadius: BorderRadius.circular(40),
                  ),
                  child: const Center(
                    child: Text(
                      'Advertisement',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 10),
                const Divider(
                  color: Colors.white,
                  thickness: 3,
                  indent: 16,
                  endIndent: 16,
                ),

                // Bottom Buttons
                Container(
                  margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      _buildBottomButton(
                        icon: Icons.wifi_off,
                        label: 'Offline mode',
                        onTap: () {
                          // Navigate to OfflinePage
                          Navigator.push(
                            context,
                            MaterialPageRoute(builder: (context) => const OfflinePage()),
                          );
                        },
                      ),
                      Container(
                        width: 70,
                        height: 70,
                        decoration: const BoxDecoration(
                          color: Colors.black,
                          shape: BoxShape.circle,
                        ),
                        child: IconButton(
                          icon: const Icon(Icons.video_call, size: 30, color: Colors.white),
                          onPressed: () {
                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (context) => const SignDetectorPage(),
                              ),
                            );
                          },
                        ),
                      ),
                      _buildBottomButton(
                        icon: Icons.chat_bubble,
                        label: 'Chat support',
                        onTap: () {
                          // Navigate to ChatSupportPage
                          Navigator.push(
                            context,
                            MaterialPageRoute(builder: (context) => const ChatSupportPage()),
                          );
                        },
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCircle({required IconData icon, required String label, VoidCallback? onTap}) {
    return Column(
      children: [
        GestureDetector(
          onTap: onTap,
          child: Container(
            width: 72,
            height: 72,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(36),
            ),
            child: Icon(icon, size: 32, color: const Color(0xFF66CDAA)),
          ),
        ),
        const SizedBox(height: 8),
        Text(
          label,
          style: const TextStyle(
            fontSize: 12,
            color: Colors.white,
          ),
        ),
      ],
    );
  }

  Widget _buildBottomButton({required IconData icon, required String label, VoidCallback? onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        children: [
          Icon(icon, size: 28, color: Colors.white),
          const SizedBox(height: 8),
          Text(
            label,
            style: const TextStyle(
              fontSize: 12,
              color: Colors.white,
            ),
          ),
        ],
      ),
    );
  }
}
