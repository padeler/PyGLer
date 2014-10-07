'''
GLUT based GUI for PyGLer
Code based on GLUMPY (https://code.google.com/p/glumpy)
@author: padeler

'''

from glutwindow import GlutWindow

GlutWindow.register_event_type('on_key_press')
GlutWindow.register_event_type('on_key_release')
GlutWindow.register_event_type('on_mouse_motion')
GlutWindow.register_event_type('on_mouse_drag')
GlutWindow.register_event_type('on_mouse_press')
GlutWindow.register_event_type('on_mouse_release')
GlutWindow.register_event_type('on_mouse_scroll')
GlutWindow.register_event_type('on_mouse_enter')
GlutWindow.register_event_type('on_mouse_leave')
GlutWindow.register_event_type('on_init')
GlutWindow.register_event_type('on_show')
GlutWindow.register_event_type('on_hide')
GlutWindow.register_event_type('on_resize')
GlutWindow.register_event_type('on_draw')
GlutWindow.register_event_type('on_idle')