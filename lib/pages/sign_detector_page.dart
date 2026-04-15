import 'dart:async';
import 'dart:convert';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class SignDetectorPage extends StatefulWidget {
  const SignDetectorPage({super.key});
  @override
  State<SignDetectorPage> createState() => _SignDetectorPageState();
}

class _SignDetectorPageState extends State<SignDetectorPage> {
  CameraController? _controller;
  List<CameraDescription> _cameras = [];
  bool _isDetecting = false;
  bool _isTakingPicture = false;
  String _detectedSign = '';
  String _sentence = '';
  List<String> _sentenceList = [];
  Timer? _timer;

  final String serverUrl = 'http://10.18.133.97:5000';

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    _cameras = await availableCameras();
    _controller = CameraController(
      _cameras[0],
      ResolutionPreset.low,  // medium se low karo
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    await _controller!.initialize();
    // Flash disable karo
    await _controller!.setFlashMode(FlashMode.off);
    // Autofocus band karo
    await _controller!.setFocusMode(FocusMode.locked);
    if (mounted) setState(() {});
  }

  void _startDetection() {
    setState(() => _isDetecting = true);
    http.post(Uri.parse('$serverUrl/reset'));

    _timer = Timer.periodic(const Duration(milliseconds: 500), (_) async {
      if (!_isDetecting || _isTakingPicture) return;
      _isTakingPicture = true;
      try {
        final image = await _controller!.takePicture();
        final bytes = await image.readAsBytes();
        final base64Image = base64Encode(bytes);

        final response = await http.post(
          Uri.parse('$serverUrl/predict'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'frame': base64Image}),
        ).timeout(const Duration(seconds: 5));

        if (response.statusCode == 200) {
          final data = jsonDecode(response.body);
          final sign = data['sign'] ?? '';
          if (sign.isNotEmpty && sign != _detectedSign) {
            setState(() {
              _detectedSign = sign;
              _sentenceList.add(sign);
              if (_sentenceList.length > 5) {
                _sentenceList = _sentenceList.sublist(_sentenceList.length - 5);
              }
              _sentence = _sentenceList.join(' ');
            });
          }
        }
      } catch (e) {
        debugPrint('Error: $e');
      } finally {
        _isTakingPicture = false;
      }
    });
  }

  void _stopDetection() {
    _timer?.cancel();
    setState(() => _isDetecting = false);
  }

  @override
  void dispose() {
    _timer?.cancel();
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D3B2E),
      appBar: AppBar(
        backgroundColor: const Color(0xFF0D3B2E),
        title: const Text('Sign Detector',
            style: TextStyle(color: Colors.white)),
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: Column(
        children: [
          Expanded(
            flex: 3,
            child: _controller != null && _controller!.value.isInitialized
                ? ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: CameraPreview(_controller!),
            )
                : const Center(
                child: CircularProgressIndicator(color: Colors.white)),
          ),
          Container(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                Text(
                  _detectedSign.isEmpty ? 'Waiting...' : _detectedSign,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 32,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  _sentence,
                  style: const TextStyle(color: Colors.white70, fontSize: 16),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: _isDetecting ? null : _startDetection,
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Start'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(
                        horizontal: 24, vertical: 12),
                  ),
                ),
                ElevatedButton.icon(
                  onPressed: _isDetecting ? _stopDetection : null,
                  icon: const Icon(Icons.stop),
                  label: const Text('Stop'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(
                        horizontal: 24, vertical: 12),
                  ),
                ),
                ElevatedButton.icon(
                  onPressed: () {
                    setState(() {
                      _sentenceList.clear();
                      _sentence = '';
                      _detectedSign = '';
                    });
                    http.post(Uri.parse('$serverUrl/reset'));
                  },
                  icon: const Icon(Icons.refresh),
                  label: const Text('Clear'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.orange,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(
                        horizontal: 24, vertical: 12),
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