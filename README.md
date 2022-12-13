#   S e n t i z a r d  
  
 T w e e t   s e n t i m e n t   a n a l y s i s   w i t h   m a c h i n e   l e a r n i n g   a n d   d e e p   l e a r n i n g   m e t h o d s  
  
 # # #   P r e r e q u i s i t e  
  
 -   I n s t a l l   s c i k i t - l e a r n ,   T e n s o r f l o w   a n d   K e r a s  
 -   G i t   c l o n e   t h i s   p r o j e c t  
 -   D o w n l o a d   d a t a s e t   f r o m   [ S e n t i m e n t 1 4 0   d a t a s e t   w i t h   1 . 6   m i l l i o n   t w e e t s ] ( h t t p s : / / w w w . k a g g l e . c o m / d a t a s e t s / k a z a n o v a / s e n t i m e n t 1 4 0 ) ,   a n d   p u t   t h e   c s v   f i l e   u n d e r   < u > Y O U R   P A T H \ T w i t t e r - S e n t i m e n t - A n a l y s i s \ B a c k e n d \ c o n t e n t \ d a t a s e t \  
  
 # #   T r a i n i n g  
  
 T h e   m o d e l   u s e d   t o   a n a l y z e   t w i t t e r   s e n t i m e n t   a r e   B e r n o u l l i N B ,   S V M ,   L S T M ,   C - L S T M   a n d   B i L S T M - C N N .  
  
 T o   t r a i n   t h e   m o d e l s ,   p l e a s e   r u n   w i t h   ` p y t h o n   n n _ t r a i n i n g . p y `   o r   ` p y t h o n   m l _ t r a i n i n g . p y ` .  
  
 # #   R e s u l t s  
  
 |   V e c t o r i z i n g   M e t h o d   |   M o d e l                                   |   A c c u r a c y   |  
 |   - - - - - - - - - - - - - - - - - -   |   - - - - - - - - - - - - - - - - - - - - -   |   - - - - - - - -   |  
 |   T F - I D F                           |   B e r n o u l l i   N a i v e   B a y e s   |   0 . 8 0           |  
 |   T F - I D F                           |   S V M                                       |   0 . 8 2           |  
 |   W o r d 2 V e c                       |   B e r n o u l l i   N a i v e   B a y e s   |   0 . 5 4           |  
 |   W o r d 2 V e c                       |   S V M                                       |   0 . 6 9           |  
 |   W o r d 2 V e c                       |   L S T M                                     |   0 . 8 2 8         |  
 |   W o r d 2 V e c                       |   C - L S T M                                 |   0 . 8 3 6         |  
 |   W o r d 2 V e c                       |   B i - L S T M                               |   0 . 8 4 6         |  
  
 