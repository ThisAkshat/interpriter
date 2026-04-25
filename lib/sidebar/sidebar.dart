import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:interpriter/pages/myaccountspage.dart';
import 'package:interpriter/pages/myorderpage.dart';
import '../Screens/signin_screen.dart';
import '../bloc.navigation_bloc/navigation_bloc.dart';
import '../sidebar/menu_item.dart';
import 'package:rxdart/rxdart.dart';


class SideBar extends StatefulWidget {
  const SideBar({super.key});

  @override
  State<SideBar> createState() => _SideBarState();
}

class _SideBarState extends State<SideBar> with SingleTickerProviderStateMixin<SideBar> {
  late AnimationController _animationController;
  late StreamController<bool> isSidebarOpenedStreamController;
  late Stream<bool> isSidebarOpenedStream;
  late StreamSink<bool> isSidebarOpenedSink;
  final _animationDuration = const Duration(milliseconds: 500);

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(vsync: this, duration: _animationDuration);
    isSidebarOpenedStreamController = PublishSubject<bool>();
    isSidebarOpenedStream = isSidebarOpenedStreamController.stream;
    isSidebarOpenedSink = isSidebarOpenedStreamController.sink;
  }

  @override
  void dispose() {
    _animationController.dispose();
    isSidebarOpenedStreamController.close();
    super.dispose();
  }

  void onIconPressed() {
    final animationStatus = _animationController.status;
    final isAnimationCompleted = animationStatus == AnimationStatus.completed;

    if (isAnimationCompleted) {
      isSidebarOpenedSink.add(false);
      _animationController.reverse();
    } else {
      isSidebarOpenedSink.add(true);
      _animationController.forward();
    }
  }

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;

    return StreamBuilder<bool>(
      initialData: false,
      stream: isSidebarOpenedStream,
      builder: (context, isSideBarOpenedAsync) {
        return AnimatedPositioned(
          duration: _animationDuration,
          top: 0,
          bottom: 0,
          left: isSideBarOpenedAsync.data == true ? 0 : -screenWidth,
          right: isSideBarOpenedAsync.data == true ? 0 : screenWidth - 34,
          child: Row(
            children: <Widget>[
              Expanded(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  color: const Color(0xFF2B4B4B),
                  child: Column(
                    children: <Widget>[
                      const SizedBox(
                        height: 100,
                      ),
                      const ListTile(
                        title: Text(
                          "Akshat",
                          style: TextStyle(color: Colors.white, fontSize: 30, fontWeight: FontWeight.w800),
                        ),
                        subtitle: Text(
                          "ak@gmail.com",
                          style: TextStyle(
                            color: Color(0xFF1BB5FD),
                            fontSize: 20,
                          ),
                        ),
                        leading: CircleAvatar(
                          radius: 40,
                          child: Icon(
                            Icons.person_sharp,
                            color: Colors.black,
                          ),
                        ),
                      ),
                      Divider(
                        height: 64,
                        thickness: 0.5,
                        color: Colors.white.withOpacity(0.3),
                        indent: 32,
                        endIndent: 32,
                      ),
                      MenuItem(
                        icon: Icons.home,
                        title: "Home",
                        onTap: () {
                          onIconPressed();
                          BlocProvider.of<NavigationBloc>(context).add(NavigationEvents.HomePageCLickedEvent);
                        },
                      ),
                      MenuItem(
                        icon: Icons.notifications,
                        title: "Notification",
                        onTap: () {
                          onIconPressed();
                          BlocProvider.of<NavigationBloc>(context).add(NavigationEvents.MyAccountClickedEvent);
                        },
                      ),
                      MenuItem(
                        icon: Icons.g_translate,
                        title: "Select Language",
                        onTap: () {
                          onIconPressed(); // Closes the sidebar
                          Navigator.push(
                            context,
                            MaterialPageRoute(builder: (context) => const MyOrderPage()),
                          );
                        },
                      ),
                      MenuItem(
                        icon: Icons.reviews,
                        title: "Reviews",
                        onTap: () {
                          onIconPressed(); // Closes the sidebar
                          Navigator.push(
                            context,
                            MaterialPageRoute(builder: (context) => const MyAccountsPage()),
                          );
                        },
                      ),
                      Divider(
                        height: 64,
                        thickness: 0.5,
                        color: Colors.white.withOpacity(0.3),
                        indent: 32,
                        endIndent: 32,
                      ),
                      MenuItem(
                        icon: Icons.settings,
                        title: "Settings",
                        onTap: () {
                          onIconPressed();
                          BlocProvider.of<NavigationBloc>(context).add(NavigationEvents.HomePageCLickedEvent);
                        },
                      ),
                      MenuItem(
                        icon: Icons.logout,
                        title: "LogOut",
                        onTap: () {
                          Navigator.pushReplacement(
                            context,
                            MaterialPageRoute(builder: (context) => const SignInScreen()),
                          );
                        },
                      ),
                    ],
                  ),
                ),
              ),
              Align(
                alignment: const Alignment(0, -
                0.9),
                child: GestureDetector(
                  onTap: () {
                    onIconPressed();
                  },
                  child: AnimatedIcon(
                    progress: _animationController.view,
                    icon: AnimatedIcons.menu_close,
                    color: Colors.white,
                    size: 35,
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}
