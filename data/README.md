version 2

删掉的字符

~ ~ ~         ->       无
\ \ \ \      ->       无
\; \; \;      ->       无
f r a c        ->      \frac
\\operatorname \{ ([a-z]+) ([a-z]+) ([a-z]+) \}             ->         \\$1$2$3
 { \kern 1 p t }      ->       无
 \\hspace { [ 0-9\-]+ p t }       ->       无
  \\hspace { [ 0-9.\-a-z]+ }
 ule \{ [ 0-9a-z.-]+ \} \{ [ 0-9a-z.-]+ \}
 \hrule \vrule 相关     ->     不要那条公式