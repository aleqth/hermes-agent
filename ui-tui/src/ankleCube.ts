import type { ThemeColors } from './theme.js'

type Line = [string, string]

// 4 unique yaw poses (11, 34, 56, 79 deg). A cube is 90-deg periodic, so
// looping these is a full visual spin without collapsing to a flat box.
export const ANKLE_CUBE_FRAMES: readonly (readonly string[])[] = [
  [
    '       -----+   ',
    ' +-----     |\\  ',
    ' |          | + ',
    ' |          | | ',
    ' |          | | ',
    ' +----------+ | ',
    '  \\          \\| ',
    '   +----------+ ',
  ],
  [
    '     ////+\\     ',
    '+////    | \\\\   ',
    '|        |    \\+',
    '|        |     |',
    '|    ////+\\    |',
    '+\\///      \\\\ | ',
    '  \\\\      ////+ ',
    '     \\+////     ',
  ],
  [
    '     /+\\\\       ',
    '  /// |    \\\\+  ',
    '+/    |        |',
    '|     |        |',
    '|    /+\\\\      |',
    '| ///      \\/+  ',
    '+\\\\        ///  ',
    '     \\\\+/       ',
  ],
  [
    '   +-----       ',
    '  /|     -----+ ',
    ' + |          | ',
    ' | |          | ',
    ' | |          | ',
    ' | +----------+ ',
    ' |/          /  ',
    ' +----------+   ',
  ],
]

export const ANKLE_CUBE_TICK_MS = 180
export const ANKLE_CUBE_WIDTH = Math.max(...ANKLE_CUBE_FRAMES.flatMap(f => f.map(l => l.length)))
export const ANKLE_CUBE_HEIGHT = ANKLE_CUBE_FRAMES[0]!.length

export const isAnkleBrand = (name: string | undefined): boolean =>
  (name ?? '').replace(/\s+/g, ' ').trim().toUpperCase() === 'ANKLE OS'

const edgeColor = (line: string, c: ThemeColors): string => {
  if (line.includes('/') || line.includes('\\')) {
    return c.accent
  }

  return c.primary
}

export function ankleCubeLines(tick: number, c: ThemeColors): Line[] {
  const frame = ANKLE_CUBE_FRAMES[tick % ANKLE_CUBE_FRAMES.length] ?? ANKLE_CUBE_FRAMES[0]!

  return frame.map(text => [edgeColor(text, c), text])
}

export function ankleWindowLines(tick: number, c: ThemeColors): Line[] {
  const cube = ANKLE_CUBE_FRAMES[tick % ANKLE_CUBE_FRAMES.length] ?? ANKLE_CUBE_FRAMES[0]!
  const inner = 42
  const pad = (s: string) => s.padEnd(inner, ' ')
  const navy = c.primary
  const ink = c.text
  const dim = c.muted
  const bar = c.accent
  const chrome = c.border
  const titleText = ' A N K L E   O S  terminal 96'.padEnd(inner, ' ')
  const out: Line[] = [
    [bar, ` ANKLE OS  [_][o][x]`.padEnd(inner + 2, ' ')],
    [chrome, `+${'-'.repeat(inner)}+`],
    [navy, `|${titleText}|`],
    [ink, `|${pad('')}|`]
  ]

  for (const row of cube) {
    out.push([edgeColor(row, c), `|${pad(`    ${row}`)}|`])
  }

  out.push([ink, `|${pad('')}|`])
  out.push([dim, `|${pad('    spinning artifact  -> type to boot')}|`])
  out.push([chrome, `+${'-'.repeat(inner)}+`])

  return out
}
