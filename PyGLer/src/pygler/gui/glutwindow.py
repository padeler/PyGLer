'''
Created on Mar 24, 2014

Uses GLUT to draw a window and handle user input.
 
Based on GLUT Window code from GLUMPY (https://code.google.com/p/glumpy)

glumpy is an OpenGL framework for the fast visualization of numpy arrays.
Copyright (C) 2009-2011  Nicolas P. Rougier. All rights reserved.

@author: padeler
'''

import OpenGL.GLUT as glut
from OpenGL import GL
import key
import mouse
import event

class GlutWindow(event.EventDispatcher):
    '''
    classdocs
    '''


    def __init__(self, size=None, position=None, title=None, fullscreen=False):
        '''
        Constructor
        '''
        self._mouse_x = 0
        self._mouse_y = 0
        self._button = mouse.NONE
        self._modifiers = None
        self._motionEventCounter=0
        self._time = None
        self._timer_stack = []
        self._timer_date = []
        self._title = title or "PyGLer" # FIXME: get default name from a central location 
        self._fullscreen = -1
        
        self.dragSensitivity = 5
        
        # Is there any glut loop already running ?
        if glut.glutGetWindow( ) == 0:
            glut.glutInit()
            glut.glutInitDisplayMode( glut.GLUT_DOUBLE |
                                      glut.GLUT_RGBA   |
                                      glut.GLUT_DEPTH )
            self._interactive = False
        else:
            self._interactive = True
            
        self._id = glut.glutCreateWindow( self._title )
        glut.glutShowWindow()        
        
        glut.glutDisplayFunc( self._display )
        glut.glutReshapeFunc( self._reshape )
        glut.glutKeyboardFunc( self._keyboard )
        glut.glutKeyboardUpFunc( self._keyboard_up )
        glut.glutMouseFunc( self._mouse )
        glut.glutMotionFunc( self._motion )
        glut.glutPassiveMotionFunc( self._passive_motion )
        glut.glutVisibilityFunc( self._visibility )
        glut.glutEntryFunc( self._entry )
        glut.glutSpecialFunc( self._special )
        glut.glutSpecialUpFunc( self._special_up )   
            
        GL.glEnable(GL.GL_DEPTH_TEST)
        
        GL.glPolygonOffset(1, 1);
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL);
          

        GL.glPointSize(5)
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
            
        if size is not None:
            width, height = size
            glut.glutReshapeWindow( width, height )

        width = glut.glutGet( glut.GLUT_WINDOW_WIDTH )
        height= glut.glutGet( glut.GLUT_WINDOW_HEIGHT )
        self._width = width
        self._height = height
        if position is not None:
            x,y = position
            glut.glutPositionWindow( x, y )
            
        x = glut.glutGet( glut.GLUT_WINDOW_X )
        y = glut.glutGet( glut.GLUT_WINDOW_X )
        self._x, self._y = x, y

        # These ones will be used when exiting fullscreen
        self._saved_width  = self._width
        self._saved_height = self._height
        self._saved_x = self._x
        self._saved_y = self._y

        self._time = glut.glutGet( glut.GLUT_ELAPSED_TIME )
        self._fullscreen = fullscreen
        if fullscreen:
            self.set_fullscreen(True)
        
# Event handler methods (registered in init)


    
    def _keyboard( self, code, x, y ):
        symbol = self._keyboard_translate(code)
        modifiers = glut.glutGetModifiers()
        modifiers = self._modifiers_translate(modifiers)
        state= self.dispatch_event('on_key_press', symbol, modifiers)
        if not state and symbol == key.ESCAPE: #FIXME 
            raise Exception," FIX EXIT MODE " #sys.exit()

    def _keyboard_up( self, code, x, y ):
        modifiers = glut.glutGetModifiers()
        self.dispatch_event('on_key_release',
                            self._keyboard_translate(code),
                            self._modifiers_translate(modifiers))

    def _special( self, code, x, y ):
        modifiers = glut.glutGetModifiers()
        self.dispatch_event('on_key_press',
                            self._keyboard_translate(code),
                            self._modifiers_translate(modifiers))

    def _special_up( self, code, x, y ):
        modifiers = glut.glutGetModifiers()
        self.dispatch_event('on_key_release',
                            self._keyboard_translate(code),
                            self._modifiers_translate(modifiers))


    def _modifiers_translate( self, modifiers ):
        _modifiers = 0
        if modifiers & glut.GLUT_ACTIVE_SHIFT:
            _modifiers |=  key.MOD_SHIFT
        if modifiers & glut.GLUT_ACTIVE_CTRL:
            _modifiers |=  key.MOD_CTRL
        if modifiers & glut.GLUT_ACTIVE_ALT:
            _modifiers |=  key.MOD_ALT
        return _modifiers


    def _keyboard_translate( self, code ):
        ascii = code
        if isinstance(code,int)==False:
            ascii = ord(code.lower())
            
        if ascii < 0x020:
            if   ascii == 0x008: return key.BACKSPACE
            elif ascii == 0x009: return key.TAB
            elif ascii == 0x00A: return key.LINEFEED
            elif ascii == 0x00C: return key.CLEAR
            elif ascii == 0x00D: return key.RETURN
            elif ascii == 0x018: return key.CANCEL
            elif ascii == 0x01B: return key.ESCAPE
            return ascii
        elif code==glut.GLUT_KEY_F1:       return key.F1
        elif code==glut.GLUT_KEY_F2:       return key.F2
        elif code==glut.GLUT_KEY_F3:       return key.F3
        elif code==glut.GLUT_KEY_F4:       return key.F4
        elif code==glut.GLUT_KEY_F5:       return key.F5
        elif code==glut.GLUT_KEY_F6:       return key.F6
        elif code==glut.GLUT_KEY_F7:       return key.F7
        elif code==glut.GLUT_KEY_F8:       return key.F8
        elif code==glut.GLUT_KEY_F9:       return key.F9
        elif code==glut.GLUT_KEY_F10:      return key.F10
        elif code==glut.GLUT_KEY_F11:      return key.F11
        elif code==glut.GLUT_KEY_F12:      return key.F12
        elif code==glut.GLUT_KEY_LEFT:     return key.LEFT
        elif code==glut.GLUT_KEY_UP:       return key.UP
        elif code==glut.GLUT_KEY_RIGHT:    return key.RIGHT
        elif code==glut.GLUT_KEY_DOWN:     return key.DOWN
        elif code==glut.GLUT_KEY_PAGE_UP:  return key.PAGEUP
        elif code==glut.GLUT_KEY_PAGE_DOWN:return key.PAGEDOWN
        elif code==glut.GLUT_KEY_HOME:     return key.HOME
        elif code==glut.GLUT_KEY_END:      return key.END
        elif code==glut.GLUT_KEY_INSERT:   return key.INSERT
        
        return ascii


    def _display( self ):
        self.dispatch_event('on_draw')
        glut.glutSwapBuffers()


    def _idle(self):
        t = glut.glutGet(glut.GLUT_ELAPSED_TIME)
        dt = (t - self._time)/1000.0
        self._time = t
        self.dispatch_event('on_idle', dt)


    def _reshape(self, width, height):
        self._width  = glut.glutGet(glut.GLUT_WINDOW_WIDTH)
        self._height = glut.glutGet(glut.GLUT_WINDOW_HEIGHT)
        self.dispatch_event('on_resize', self._width, self._height)


    def _visibility(self, state):
        if state == glut.GLUT_VISIBLE:
            self.dispatch_event('on_show')
        elif state == glut.GLUT_NOT_VISIBLE:
            self.dispatch_event('on_hide')

    def _entry(self, state):
        if state == glut.GLUT_ENTERED:
            self.dispatch_event('on_mouse_enter')
        elif state == glut.GLUT_LEFT:
            self.dispatch_event('on_mouse_leave')


    def _mouse(self, button, state, x, y):
        y = self._height - y
        if button == glut.GLUT_LEFT_BUTTON:
            button = mouse.LEFT
        elif button == glut.GLUT_MIDDLE_BUTTON:
            button = mouse.MIDDLE
        elif button == glut.GLUT_RIGHT_BUTTON:
            button = mouse.RIGHT
            
        if state == glut.GLUT_UP:
            self._button = mouse.NONE
            self._mouse_x = x
            self._mouse_y = y
            self.dispatch_event('on_mouse_release', x, y, button)
        elif state == glut.GLUT_DOWN:
            self._button = button
            self._mouse_x = x
            self._mouse_y = y
            if button == 3:
                self.dispatch_event('on_mouse_scroll', x, y, 0, 1)
            elif button == 4:
                self.dispatch_event('on_mouse_scroll', x, y, 0, -1)
            else:
                self.dispatch_event('on_mouse_press', x, y, button)


    def _motion(self, x, y):
        y = self._height - y
        # dispatch events sparsely.
        if self._motionEventCounter%self.dragSensitivity==0:             
            dx = x - self._mouse_x
            dy = y - self._mouse_y
            self._mouse_x = x
            self._mouse_y = y
            self.dispatch_event('on_mouse_drag', x, y, dx, dy, self._button)

        self._motionEventCounter+=1
            


    def _passive_motion(self, x, y):
        y = self._height - y

        # dispatch events sparsely.
        if self._motionEventCounter%self.dragSensitivity==0:             
            dx = x - self._mouse_x
            dy = y - self._mouse_y
            self._mouse_x = x
            self._mouse_y = y
            self.dispatch_event('on_mouse_motion', x, y, dx, dy)

        self._motionEventCounter+=1


    def show(self):
        '''
        The show() method causes a window to be displayed as soon as possible.
        '''

        glut.glutSetWindow( self._id )
        glut.glutShowWindow()
        self.dispatch_event('on_show')


    def hide(self):
        '''
        The hide() method reverses the effects of the show() method, causing
        the window to be hidden (removed from the display).
        '''

        glut.glutSetWindow( self._id )
        glut.glutHideWindow()
        self.dispatch_event('on_hide')


    def redraw(self):
        '''
        The redraw() method invalidates the window area. Once the main loop
        becomes idle (after the current batch of events has been processed,
        roughly), the window will dispatch a ``draw`` event and swaps the
        buffers if double buffered.
        '''

        self.dispatch_event('on_draw')
        glut.glutSwapBuffers()
        
        
    def refresh(self):
        ''' Refresh the window content by swapping back and fron buffer.. '''
        glut.glutSwapBuffers()


    def setFullscreen(self, state):
        '''
        If **state** is True, the set_fullscreen() method requests the window
        manager to place the window in the fullscreen state. If **state** is
        False the set_fullscreen() method requests the window manager to toggle
        off the fullscreen state for the window. Note that in any case, you
        shouldn't not assume the window state is definitely set afterward,
        because other entities (e.g. the user or window manager) could
        fullscreen/unfullscreen it again, and not all window managers honor
        requests to fullscreen windows.

        :param bool state:
            Fullscreen state to be set.
        '''

        if self._fullscreen == state:
            return

        if state == True:
            glut.glutSetWindow( self._id )
            self._saved_width  = glut.glutGet(glut.GLUT_WINDOW_WIDTH)
            self._saved_height = glut.glutGet(glut.GLUT_WINDOW_HEIGHT)
            self._saved_x = glut.glutGet(glut.GLUT_WINDOW_X)
            self._saved_y = glut.glutGet(glut.GLUT_WINDOW_Y)
            self._fullscreen = True
            glut.glutFullScreen()
        else:
            self._fullscreen = False
            glut.glutSetWindow( self._id )
            glut.glutReshapeWindow(self._saved_width, self._saved_height)
            glut.glutPositionWindow( self._saved_x, self._saved_y )
            glut.glutSetWindowTitle( self._title )
            
    def getFullscreen(self):
        '''
        Return whether window is currently in fullscreen state or not.
        '''

        return self._fullscreen


    def setTitle(self, title):
        '''
        The set_title() method sets the "title" property of the Window to the
        value specified by title. The title of a window will be displayed in
        its title bar. On the X Window System, the title bar is rendered by the
        window manager, so exactly how the title appears to users may vary
        according to a user's exact configuration. The title should help a user
        distinguish this window from other windows they may have open. A good
        title might include the application name and current document filename.

        :param string title:
            the title of the window.
        '''

        glut.glutSetWindow( self._id )
        glut.glutSetWindowTitle( title )
        self._title = title


    def getTitle(self, title):
        '''

        The get_title() method returns the value of the "title" property of the
        window. See the set_title() method.

        :rtype: str
        :return: the title of the window
        '''

        return self._title


    def setSize(self, width, height):
        '''
        The set_size() method requests the window manager to resize the window
        to the specified width and height as if the user had done so, obeying
        geometry constraints. Note you shouldn't assume the new window size is
        definitely the requested one afterward, because other entities (e.g. the user
        or window manager) could change it ssize again, and not all window
        managers honor requests to resize windows. 

        :param integer width:
            The new width of the window, in pixels.

        :param integer height:
            The new height of the window, in pixels.
        '''

        glut.glutReshapeWindow(width, height)


    def getSize(self):
        '''
        The get_size() methods returns the current size of the window and does
        not include the size of the window manager decorations (aka the window
        frame or border). 

        :rtype: (int, int)
        :return: The width and height of the window, in pixels.
        '''

        glut.glutSetWindow( self._id )
        self._width  = glut.glutGet( glut.GLUT_WINDOW_WIDTH )
        self._height = glut.glutGet( glut.GLUT_WINDOW_HEIGHT )
        return self._width, self._height


    def setPosition(self, x, y):
        '''
        The set_position() method requests the window manager to move the
        window to the specified coordinates as if the user had done so, obeying
        geometry constraints. Note you shouldn't assume the new window position
        is definitely the requested one afterward, because other entities
        (e.g. the user or window manager) could change it position again, and
        not all window managers honor requests to move windows.

        :param integer x:
            The x coordinate in pixels to move the window to

        :param integer y:
            The y coordinate in pixels to move the window to
        '''

        glut.glutPositionWindow( x, y )


    def getPosition(self):
        '''
        The get_position() method returns the current posiiton of the window. 

        :rtype: (int, int)
        :return: The current window coordinates, in pixels.
        '''

        glut.glutSetWindow( self._id )
        self._x = glut.glutGet( glut.GLUT_WINDOW_X )
        self._y = glut.glutGet( glut.GLUT_WINDOW_Y )
        return self._x, self._y



    def start(self):
        ''' Starts main loop. '''

        # Start timers
        for i in range(len(self._timer_stack)):
            def func(index):
                handler, fps = self._timer_stack[index]
                t = glut.glutGet(glut.GLUT_ELAPSED_TIME)
                dt = (t - self._timer_date[index])/1000.0
                self._timer_date[index] = t
                handler(dt)
                glut.glutTimerFunc(int(1000./fps), func, index)
                self._timer_date[index] = glut.glutGet(glut.GLUT_ELAPSED_TIME)
            fps = self._timer_stack[i][1]
            glut.glutTimerFunc(int(1000./fps), func, i)

        # Start idle only if necessary
        for item in self._event_stack:
            if 'on_idle' in item.keys():
                glut.glutIdleFunc(self._idle)

        # Dispatch init event
        self.dispatch_event('on_init')
        glut.glutMainLoop()
        

    def stop(self):
        '''Exit mainloop'''
        # FIXME This will also kill the interpreter. 
        # we do not want that. just to close the window and release resources.
        if (glut.glutLeaveMainLoop):
            glut.glutLeaveMainLoop()
        else:
            raise RuntimeError("Your GLUT implementation does not allow to stops the main loop")


