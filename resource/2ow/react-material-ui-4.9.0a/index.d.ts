declare namespace MaterialUI {
// disable automatic export
/**
 * `T extends ConsistentWith<T, U>` means that where `T` has overlapping properties with
 * `U`, their value types do not conflict.
 *
 * @internal
 */
export type ConsistentWith<DecorationTargetProps, InjectedProps> = {
  [P in keyof DecorationTargetProps]: P extends keyof InjectedProps
    ? InjectedProps[P] extends DecorationTargetProps[P]
      ? DecorationTargetProps[P]
      : InjectedProps[P]
    : DecorationTargetProps[P]
};
/**
 * a function that takes {component} and returns a component that passes along
 * all the props to {component} except the {InjectedProps} and will accept
 * additional {AdditionalProps}
 */
export type PropInjector<InjectedProps, AdditionalProps = {}> = <
  C extends React.ComponentType<ConsistentWith<React.ComponentProps<C>, InjectedProps>>
>(
  component: C,
) => React.ComponentType<
  Omit<JSX.LibraryManagedAttributes<C, React.ComponentProps<C>>, keyof InjectedProps> &
    AdditionalProps
>;
/**
 * Remove properties `K` from `T`.
 *
 * @internal
 */
export type Omit<T, K extends keyof any> = T extends any ? Pick<T, Exclude<keyof T, K>> : never;
/**
 * Like `T & U`, but using the value types from `U` where their properties overlap.
 *
 * @internal
 */
export type Overwrite<T, U> = Omit<T, keyof U> & U;
}
declare namespace MaterialUI {
/**
 * @deprecated
 * Import from `@material-ui/types` instead
 *
 * TODO: to remove in v5
 */
/**
 * All standard components exposed by `material-ui` are `StyledComponents` with
 * certain `classes`, on which one can also set a top-level `className` and inline
 * `style`.
 */
export type StandardProps<C, ClassKey extends string, Removals extends keyof C = never> = Omit<
  C,
  'classes' | Removals
> &
  StyledComponentProps<ClassKey> & {
    className?: string;
  key?: string;
    ref?: C extends { ref?: infer RefType } ? RefType : React.Ref<unknown>;
    style?: React.CSSProperties;
  };
export type PaletteType = 'light' | 'dark';
export interface Color {
  50: string;
  100: string;
  200: string;
  300: string;
  400: string;
  500: string;
  600: string;
  700: string;
  800: string;
  900: string;
  A100: string;
  A200: string;
  A400: string;
  A700: string;
}
export namespace PropTypes {
  type Alignment = 'inherit' | 'left' | 'center' | 'right' | 'justify';
  type Color = 'inherit' | 'primary' | 'secondary' | 'default';
  type Margin = 'none' | 'dense' | 'normal';
}
// From index.js
}
declare namespace MaterialUI {
export interface PaperProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, PaperClassKey> {
  component?: React.ElementType<React.HTMLAttributes<HTMLElement>>;
  elevation?: number;
  square?: boolean;
  variant?: 'elevation' | 'outlined';
}
export type PaperClassKey =
  | 'root'
  | 'rounded'
  | 'outlined'
  | 'elevation0'
  | 'elevation1'
  | 'elevation2'
  | 'elevation3'
  | 'elevation4'
  | 'elevation5'
  | 'elevation6'
  | 'elevation7'
  | 'elevation8'
  | 'elevation9'
  | 'elevation10'
  | 'elevation11'
  | 'elevation12'
  | 'elevation13'
  | 'elevation14'
  | 'elevation15'
  | 'elevation16'
  | 'elevation17'
  | 'elevation18'
  | 'elevation19'
  | 'elevation20'
  | 'elevation21'
  | 'elevation22'
  | 'elevation23'
  | 'elevation24';
export const Paper: React.ComponentType<PaperProps>;
//export default Paper;
}
declare namespace MaterialUI {
export interface AppBarProps extends StandardProps<PaperProps, AppBarClassKey> {
  /**
   * The color of the component. It supports those theme colors that make sense for this component.
   */
  color?: PropTypes.Color;
  /**
   * The positioning type. The behavior of the different options is described
   * [in the MDN web docs](https://developer.mozilla.org/en-US/docs/Learn/CSS/CSS_layout/Positioning).
   * Note: `sticky` is not universally supported and will fall back to `static` when unavailable.
   */
  position?: 'fixed' | 'absolute' | 'sticky' | 'static' | 'relative';
}
export type AppBarClassKey =
  | 'root'
  | 'positionFixed'
  | 'positionAbsolute'
  | 'positionSticky'
  | 'positionStatic'
  | 'positionRelative'
  | 'colorDefault'
  | 'colorPrimary'
  | 'colorSecondary';
export function AppBar(props: AppBarProps): JSX.Element;
}
declare namespace MaterialUI {
/**
 * A component whose root component can be controlled via a `component` prop.
 *
 * Adjusts valid props based on the type of `component`.
 */
export interface OverridableComponent<M extends OverridableTypeMap> {
  <C extends React.ElementType>(props: { component: C } & OverrideProps<M, C>): JSX.Element;
  (props: DefaultComponentProps<M>): JSX.Element;
}
/**
 * Props of the component if `component={Component}` is used.
 */
// prettier-ignore
export type OverrideProps<
  M extends OverridableTypeMap,
  C extends React.ElementType
> = (
  & BaseProps<M>
  & Omit<React.ComponentPropsWithRef<C>, keyof CommonProps<M>>
);
/**
 * Props if `component={Component}` is NOT used.
 */
// prettier-ignore
export type DefaultComponentProps<M extends OverridableTypeMap> =
  & BaseProps<M>
  & Omit<React.ComponentPropsWithRef<M['defaultComponent']>, keyof BaseProps<M>>;
/**
 * Props defined on the component (+ common material-ui props).
 */
// prettier-ignore
export type BaseProps<M extends OverridableTypeMap> =
  & M['props']
  & CommonProps<M>;
/**
 * Props that are valid for material-ui components.
 */
export interface CommonProps<M extends OverridableTypeMap>
  extends StyledComponentProps<M['classKey']> {
  className?: string;
  key?: string;
  style?: React.CSSProperties;
}
export interface OverridableTypeMap {
  props: {};
  defaultComponent: React.ElementType;
  classKey: string;
}
/**
 * @deprecated Not used in this library.
 */
export type Simplify<T> = T extends any ? { [K in keyof T]: T[K] } : never;
/**
 * @deprecated Not used in this library.
 */
// tslint:disable-next-line: deprecation
export type SimplifiedPropsOf<C extends React.ElementType> = Simplify<React.ComponentProps<C>>;
}
declare namespace MaterialUI {
export interface AvatarTypeMap<P = {}, D extends React.ElementType = 'div'> {
  props: P & {
    alt?: string;
    imgProps?: React.ImgHTMLAttributes<HTMLImageElement>;
    sizes?: string;
    src?: string;
    srcSet?: string;
    variant?: 'circle' | 'rounded' | 'square';
  };
  defaultComponent: D;
  classKey: AvatarClassKey;
}
export const Avatar: OverridableComponent<AvatarTypeMap>;
export type AvatarClassKey =
  | 'root'
  | 'colorDefault'
  | 'circle'
  | 'rounded'
  | 'square'
  | 'img'
  | 'fallback';
export type AvatarProps<
  D extends React.ElementType = AvatarTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<AvatarTypeMap<P, D>, D>;
//export default Avatar;
}
declare namespace MaterialUI {
export type Breakpoint = 'xs' | 'sm' | 'md' | 'lg' | 'xl';
export type BreakpointValues = { [key in Breakpoint]: number };
export const keys: Breakpoint[];
export interface Breakpoints {
  keys: Breakpoint[];
  values: BreakpointValues;
  up: (key: Breakpoint | number) => string;
  down: (key: Breakpoint | number) => string;
  between: (start: Breakpoint | number, end: Breakpoint | number) => string;
  only: (key: Breakpoint) => string;
  width: (key: Breakpoint) => number;
}
export type BreakpointsOptions = Partial<
  {
    unit: string;
    step: number;
  } & Breakpoints
>;
export function createBreakpoints(options: BreakpointsOptions): Breakpoints;
}
declare namespace MaterialUI {
/* tslint:disable:unified-signatures */
export type SpacingArgument = number;
export interface Spacing {
  (): number;
  (value1: SpacingArgument): number;
  (value1: SpacingArgument, value2: SpacingArgument): string;
  (value1: SpacingArgument, value2: SpacingArgument, value3: SpacingArgument): string;
  (
    value1: SpacingArgument,
    value2: SpacingArgument,
    value3: SpacingArgument,
    value4: SpacingArgument,
  ): string;
}
export type SpacingOptions = number | ((factor: number) => string | number);
export function createSpacing(spacing: SpacingOptions): Spacing;
}
declare namespace MaterialUI {
export interface Mixins {
  gutters: (styles?: React.CSSProperties) => React.CSSProperties;
  toolbar: React.CSSProperties;
  // ... use interface declaration merging to add custom mixins
}
export interface MixinsOptions extends Partial<Mixins> {
  // ... use interface declaration merging to add custom mixin options
}
export function createMixins(
  breakpoints: Breakpoints,
  spacing: Spacing,
  mixins: MixinsOptions,
): Mixins;
}
declare namespace MaterialUI {
export interface CommonColors {
  black: string;
  white: string;
}
export const common: CommonColors;
//export default common;
}
declare namespace MaterialUI {
export type ColorPartial = Partial<Color>;
export interface TypeText {
  primary: string;
  secondary: string;
  disabled: string;
  hint: string;
}
export interface TypeAction {
  active: string;
  hover: string;
  hoverOpacity: number;
  selected: string;
  disabled: string;
  disabledBackground: string;
}
export interface TypeBackground {
  default: string;
  paper: string;
}
export type TypeDivider = string;
export type PaletteColorOptions = SimplePaletteColorOptions | ColorPartial;
export interface SimplePaletteColorOptions {
  light?: string;
  main: string;
  dark?: string;
  contrastText?: string;
}
export interface PaletteColor {
  light: string;
  main: string;
  dark: string;
  contrastText: string;
}
export interface TypeObject {
  text: TypeText;
  action: TypeAction;
  divider: TypeDivider;
  background: TypeBackground;
}
export const light: TypeObject;
export const dark: TypeObject;
export interface Palette {
  common: CommonColors;
  type: PaletteType;
  contrastThreshold: number;
  tonalOffset: number;
  primary: PaletteColor;
  secondary: PaletteColor;
  error: PaletteColor;
  warning: PaletteColor;
  info: PaletteColor;
  success: PaletteColor;
  grey: Color;
  text: TypeText;
  divider: TypeDivider;
  action: TypeAction;
  background: TypeBackground;
  getContrastText: (background: string) => string;
  augmentColor: {
    (
      color: ColorPartial,
      mainShade?: number | string,
      lightShade?: number | string,
      darkShade?: number | string,
    ): PaletteColor;
    (color: PaletteColorOptions): PaletteColor;
  };
}
export type PartialTypeObject = { [P in keyof TypeObject]?: Partial<TypeObject[P]> };
export interface PaletteOptions {
  primary?: PaletteColorOptions;
  secondary?: PaletteColorOptions;
  error?: PaletteColorOptions;
  warning?: PaletteColorOptions;
  info?: PaletteColorOptions;
  success?: PaletteColorOptions;
  type?: PaletteType;
  tonalOffset?: number;
  contrastThreshold?: number;
  common?: Partial<CommonColors>;
  grey?: ColorPartial;
  text?: Partial<TypeText>;
  divider?: string;
  action?: Partial<TypeAction>;
  background?: Partial<TypeBackground>;
  getContrastText?: (background: string) => string;
}
export function createPalette(palette: PaletteOptions): Palette;
}
declare namespace MaterialUI {
export {
  CreateCSSProperties,
  CSSProperties,
  ClassNameMap,
  StyledComponentProps,
  Styles,
  WithStylesOptions,
  StyleRulesCallback,
  BaseCSSProperties,
};
/**
 * Adapter for `StyleRules` from `@material-ui/styles` for backwards compatibility.
 * Order of generic arguments is just reversed.
 *
 * TODO: to normalize in v5
 */
export type StyleRules<
  ClassKey extends string = string,
  Props extends object = {}
> = ActualStyleRules<Props, ClassKey>;
export type WithStyles<
  StylesOrClassKey extends string | Styles<any, any, any> = string,
  IncludeTheme extends boolean | undefined = false
> = (IncludeTheme extends true ? { theme: Theme } : {}) & {
  classes: ClassNameMap<ClassKeyOfStyles<StylesOrClassKey>>;
};
export function withStyles<
  ClassKey extends string,
  Options extends WithStylesOptions<Theme> = {},
  Props extends object = {}
>(
  style: Styles<Theme, Props, ClassKey>,
  options?: Options,
): PropInjector<WithStyles<ClassKey, Options['withTheme']>, StyledComponentProps<ClassKey> & Props>;
}
declare namespace MaterialUI {
export type Variant =
  | 'h1'
  | 'h2'
  | 'h3'
  | 'h4'
  | 'h5'
  | 'h6'
  | 'subtitle1'
  | 'subtitle2'
  | 'body1'
  | 'body2'
  | 'caption'
  | 'button'
  | 'overline';
export interface FontStyle
  extends Required<{
    fontFamily: React.CSSProperties['fontFamily'];
    fontSize: number;
    fontWeightLight: React.CSSProperties['fontWeight'];
    fontWeightRegular: React.CSSProperties['fontWeight'];
    fontWeightMedium: React.CSSProperties['fontWeight'];
    fontWeightBold: React.CSSProperties['fontWeight'];
  }> {}
export interface FontStyleOptions extends Partial<FontStyle> {
  htmlFontSize?: number;
  allVariants?: React.CSSProperties;
}
// TODO: which one should actually be allowed to be subject to module augmentation?
// current type vs interface decision is kept for historical reasons until we
// made a decision
export type TypographyStyle = CSSProperties;
export interface TypographyStyleOptions extends TypographyStyle {}
export interface TypographyUtils {
  pxToRem: (px: number) => string;
}
export interface Typography extends Record<Variant, TypographyStyle>, FontStyle, TypographyUtils {}
export interface TypographyOptions
  extends Partial<Record<Variant, TypographyStyleOptions> & FontStyleOptions> {}
export function createTypography(
  palette: Palette,
  typography: TypographyOptions | ((palette: Palette) => TypographyOptions),
): Typography;
}
declare namespace MaterialUI {
export type Shadows = [
  'none',
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string
];
export const shadows: Shadows;
//export default shadows;
}
declare namespace MaterialUI {
export interface Shape {
  borderRadius: number;
}
export type ShapeOptions = Partial<Shape>;
export const shape: Shape;
//export default shape;
}
declare namespace MaterialUI {
export interface Easing {
  easeInOut: string;
  easeOut: string;
  easeIn: string;
  sharp: string;
}
export const easing: Easing;
export interface Duration {
  shortest: number;
  shorter: number;
  short: number;
  standard: number;
  complex: number;
  enteringScreen: number;
  leavingScreen: number;
}
export const duration: Duration;
export function formatMs(milliseconds: number): string;
export interface Transitions {
  easing: Easing;
  duration: Duration;
  create(
    props: string | string[],
    options?: Partial<{ duration: number | string; easing: string; delay: number | string }>,
  ): string;
  getAutoHeightDuration(height: number): number;
}
export interface TransitionsOptions {
  easing?: Partial<Easing>;
  duration?: Partial<Duration>;
  create?: (
    props: string | string[],
    options?: Partial<{ duration: number | string; easing: string; delay: number | string }>,
  ) => string;
  getAutoHeightDuration?: (height: number) => number;
}
// export type TransitionsOptions = DeepPartial<Transitions>;
export const transitions: Transitions;
//export default transitions;
}
declare namespace MaterialUI {
export interface ZIndex {
  mobileStepper: number;
  speedDial: number;
  appBar: number;
  drawer: number;
  modal: number;
  snackbar: number;
  tooltip: number;
}
export type ZIndexOptions = Partial<ZIndex>;
export const zIndex: ZIndex;
//export default zIndex;
}
declare namespace MaterialUI {
export interface BadgeOrigin {
  vertical: 'top' | 'bottom';
  horizontal: 'left' | 'right';
}
export interface BadgeProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, BadgeClassKey> {
  /**
   * The anchor of the badge.
   */
  anchorOrigin?: BadgeOrigin;
  /**
   * Wrapped shape the badge should overlap.
   */
  overlap?: 'rectangle' | 'circle';
  /**
   * The content rendered within the badge.
   */
  badgeContent?: React.ReactNode;
  /**
   * The badge will be added relative to this node.
   */
  children?: React.ReactNode;
  /**
   * The color of the component. It supports those theme colors that make sense for this component.
   */
  color?: 'primary' | 'secondary' | 'default' | 'error';
  /**
   * The component used for the root node.
   * Either a string to use a DOM element or a component.
   */
  component?: React.ElementType<React.HTMLAttributes<HTMLDivElement>>;
  /**
   * If `true`, the badge will be invisible.
   */
  invisible?: boolean;
  /**
   * Max count to show.
   */
  max?: number;
  /**
   * Controls whether the badge is hidden when `badgeContent` is zero.
   */
  showZero?: boolean;
  /**
   * The variant to use.
   */
  variant?: 'standard' | 'dot';
}
export type BadgeClassKey =
  | 'root'
  | 'badge'
  | 'colorPrimary'
  | 'colorSecondary'
  | 'colorError'
  | 'dot'
  | 'anchorOriginTopRightRectangle'
  | 'anchorOriginBottomRightRectangle'
  | 'anchorOriginTopLeftRectangle'
  | 'anchorOriginBottomLeftRectangle'
  | 'anchorOriginTopRightCircle'
  | 'anchorOriginBottomRightCircle'
  | 'anchorOriginTopLeftCircle'
  | 'invisible';
export function Badge(props: BadgeProps): JSX.Element | null;
}
declare namespace MaterialUI {
export type TouchRippleProps = StandardProps<
  React.HTMLAttributes<HTMLElement>,
  TouchRippleClassKey
> & {
  center?: boolean;
};
export type TouchRippleClassKey =
  | 'root'
  | 'ripple'
  | 'rippleVisible'
  | 'ripplePulsate'
  | 'child'
  | 'childLeaving'
  | 'childPulsate';
export const TouchRipple: React.ComponentType<TouchRippleProps>;
//export default TouchRipple;
}
declare namespace MaterialUI {
export interface ButtonBaseTypeMap<P = {}, D extends React.ElementType = 'button'> {
  props: P & {
    action?: React.Ref<ButtonBaseActions>;
    /**
     * Prefer `ref` instead.
     */
    buttonRef?: React.Ref<unknown>;
    centerRipple?: boolean;
    disabled?: boolean;
    disableRipple?: boolean;
    disableTouchRipple?: boolean;
    focusRipple?: boolean;
    focusVisibleClassName?: string;
    onFocusVisible?: React.FocusEventHandler<any>;
    TouchRippleProps?: Partial<TouchRippleProps>;
  };
  defaultComponent: D;
  classKey: ButtonBaseClassKey;
}
/**
 * utility to create component types that inherit props from ButtonBase.
 * This component has an additional overload if the `href` prop is set which
 * can make extension quite tricky
 */
export interface ExtendButtonBaseTypeMap<M extends OverridableTypeMap> {
  props: ButtonBaseTypeMap['props'] & M['props'];
  defaultComponent: M['defaultComponent'];
  classKey: M['classKey'];
}
export type ExtendButtonBase<M extends OverridableTypeMap> = ((
  props: { href: string } & OverrideProps<ExtendButtonBaseTypeMap<M>, 'a'>,
) => JSX.Element) &
  OverridableComponent<ExtendButtonBaseTypeMap<M>>;
export const ButtonBase: ExtendButtonBase<ButtonBaseTypeMap>;
export type ButtonBaseProps<
  D extends React.ElementType = ButtonBaseTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<ButtonBaseTypeMap<P, D>, D>;
export type ButtonBaseClassKey = 'root' | 'disabled' | 'focusVisible';
export interface ButtonBaseActions {
  focusVisible(): void;
}
//export default ButtonBase;
}
declare namespace MaterialUI {
export type BottomNavigationActionTypeMap<
  P,
  D extends React.ElementType
> = ExtendButtonBaseTypeMap<{
  props: P & {
    icon?: string | React.ReactElement;
    label?: React.ReactNode;
    onChange?: (event: React.ChangeEvent<{}>, value: any) => void;
    onClick?: React.ReactEventHandler<any>;
    selected?: boolean;
    showLabel?: boolean;
    value?: any;
  };
  defaultComponent: D;
  classKey: BottomNavigationActionClassKey;
}>;
export const BottomNavigationAction: ExtendButtonBase<
  BottomNavigationActionTypeMap<{}, ButtonBaseTypeMap['defaultComponent']>
>;
export type BottomNavigationActionClassKey = 'root' | 'selected' | 'iconOnly' | 'wrapper' | 'label';
export type BottomNavigationActionProps<
  D extends React.ElementType = ButtonBaseTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<BottomNavigationActionTypeMap<P, D>, D>;
//export default BottomNavigationAction;
}
declare namespace MaterialUI {
export interface BottomNavigationProps
  extends StandardProps<
    React.HTMLAttributes<HTMLDivElement>,
    BottomNavigationClassKey,
    'onChange'
  > {
  /**
   * The component used for the root node.
   * Either a string to use a DOM element or a component.
   */
  component?: React.ElementType<React.HTMLAttributes<HTMLDivElement>>;
  /**
   * Callback fired when the value changes.
   *
   * @param {object} event The event source of the callback.
   * @param {any} value We default to the index of the child.
   */
  onChange?(event: React.ChangeEvent<{}>, value: any): void;
  /**
   * If `true`, all `BottomNavigationAction`s will show their labels.
   * By default, only the selected `BottomNavigationAction` will show its label.
   */
  showLabels?: boolean;
  /**
   * The value of the currently selected `BottomNavigationAction`.
   */
  value?: any;
}
export type BottomNavigationClassKey = 'root';
export function BottomNavigation(props: BottomNavigationProps): JSX.Element;
}
declare namespace MaterialUI {
export interface BreadcrumbsTypeMap<P = {}, D extends React.ElementType = 'nav'> {
  props: P & {
    itemsAfterCollapse?: number;
    itemsBeforeCollapse?: number;
    maxItems?: number;
    separator?: React.ReactNode;
  };
  defaultComponent: D;
  classKey: BreadcrumbsClassKey;
}
export const Breadcrumbs: OverridableComponent<BreadcrumbsTypeMap>;
export type BreadcrumbsClassKey = 'root' | 'ol' | 'li' | 'separator';
export type BreadcrumbsProps<
  D extends React.ElementType = BreadcrumbsTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<BreadcrumbsTypeMap<P, D>, D>;
//export default Breadcrumbs;
}
declare namespace MaterialUI {
export type ButtonTypeMap<
  P = {},
  D extends React.ElementType = 'button'
> = ExtendButtonBaseTypeMap<{
  props: P & {
    color?: PropTypes.Color;
    disableElevation?: boolean;
    disableFocusRipple?: boolean;
    endIcon?: React.ReactNode;
    fullWidth?: boolean;
    href?: string;
    size?: 'small' | 'medium' | 'large';
    startIcon?: React.ReactNode;
    variant?: 'text' | 'outlined' | 'contained';
  };
  defaultComponent: D;
  classKey: ButtonClassKey;
}>;
export const Button: ExtendButtonBase<ButtonTypeMap>;
export type ButtonProps<
  D extends React.ElementType = ButtonTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<ButtonTypeMap<P, D>, D>;
export type ButtonClassKey =
  | 'root'
  | 'label'
  | 'text'
  | 'textPrimary'
  | 'textSecondary'
  | 'outlined'
  | 'outlinedPrimary'
  | 'outlinedSecondary'
  | 'contained'
  | 'containedPrimary'
  | 'containedSecondary'
  | 'disableElevation'
  | 'focusVisible'
  | 'disabled'
  | 'colorInherit'
  | 'textSizeSmall'
  | 'textSizeLarge'
  | 'outlinedSizeSmall'
  | 'outlinedSizeLarge'
  | 'containedSizeSmall'
  | 'containedSizeLarge'
  | 'sizeSmall'
  | 'sizeLarge'
  | 'fullWidth'
  | 'startIcon'
  | 'endIcon'
  | 'iconSizeSmall'
  | 'iconSizeMedium'
  | 'iconSizeLarge';
//export default Button;
}
declare namespace MaterialUI {
export interface ButtonGroupTypeMap<P = {}, D extends React.ElementType = 'div'> {
  props: P & {
    color?: PropTypes.Color;
    disabled?: boolean;
    disableFocusRipple?: boolean;
    disableRipple?: boolean;
    fullWidth?: boolean;
    orientation?: 'vertical' | 'horizontal';
    size?: 'small' | 'medium' | 'large';
    variant?: 'text' | 'outlined' | 'contained';
  };
  defaultComponent: D;
  classKey: ButtonGroupClassKey;
}
export const ButtonGroup: OverridableComponent<ButtonGroupTypeMap>;
export type ButtonGroupClassKey =
  | 'root'
  | 'contained'
  | 'disabled'
  | 'fullWidth'
  | 'vertical'
  | 'grouped'
  | 'groupedHorizontal'
  | 'groupedVertical'
  | 'groupedText'
  | 'groupedTextHorizontal'
  | 'groupedTextVertical'
  | 'groupedTextPrimary'
  | 'groupedTextSecondary'
  | 'groupedOutlined'
  | 'groupedOutlinedHorizontal'
  | 'groupedOutlinedVertical'
  | 'groupedOutlinedPrimary'
  | 'groupedOutlinedSecondary'
  | 'groupedContained'
  | 'groupedContainedHorizontal'
  | 'groupedContainedVertical'
  | 'groupedContainedPrimary'
  | 'groupedContainedSecondary';
export type ButtonGroupProps<
  D extends React.ElementType = ButtonGroupTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<ButtonGroupTypeMap<P, D>, D>;
//export default ButtonGroup;
}
declare namespace MaterialUI {
export type CardActionAreaTypeMap<P, D extends React.ElementType> = ExtendButtonBaseTypeMap<{
  props: P & {
    focusVisibleClassName?: string;
  };
  defaultComponent: D;
  classKey: CardActionAreaClassKey;
}>;
export const CardActionArea: ExtendButtonBase<
  CardActionAreaTypeMap<{}, ButtonBaseTypeMap['defaultComponent']>
>;
export type CardActionAreaClassKey = 'root' | 'focusVisible' | 'focusHighlight';
export type CardActionAreaProps<
  D extends React.ElementType = ButtonBaseTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<CardActionAreaTypeMap<P, D>, D>;
//export default CardActionArea;
}
declare namespace MaterialUI {
export interface CardActionsProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, CardActionsClassKey> {
  disableSpacing?: boolean;
}
export type CardActionsClassKey = 'root' | 'spacing';
export const CardActions: React.ComponentType<CardActionsProps>;
//export default CardActions;
}
declare namespace MaterialUI {
export interface CardProps extends StandardProps<PaperProps, CardClassKey> {
  raised?: boolean;
}
export type CardClassKey = 'root';
export const Card: React.ComponentType<CardProps>;
//export default Card;
}
declare namespace MaterialUI {
export interface CardContentProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, CardContentClassKey> {
  component?: React.ElementType<React.HTMLAttributes<HTMLDivElement>>;
}
export type CardContentClassKey = 'root';
export const CardContent: React.ComponentType<CardContentProps>;
//export default CardContent;
}
declare namespace MaterialUI {
type Variant = ThemeVariant | 'srOnly';
export interface TypographyTypeMap<P = {}, D extends React.ElementType = 'span'> {
  props: P & {
    align?: PropTypes.Alignment;
    color?:
      | 'initial'
      | 'inherit'
      | 'primary'
      | 'secondary'
      | 'textPrimary'
      | 'textSecondary'
      | 'error';
    display?: 'initial' | 'block' | 'inline';
    gutterBottom?: boolean;
    noWrap?: boolean;
    paragraph?: boolean;
    variant?: Variant | 'inherit';
    variantMapping?: Partial<Record<Variant, string>>;
  };
  defaultComponent: D;
  classKey: TypographyClassKey;
}
export const Typography: OverridableComponent<TypographyTypeMap>;
export type TypographyProps<
  D extends React.ElementType = TypographyTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<TypographyTypeMap<P, D>, D>;
export type TypographyClassKey =
  | 'root'
  | 'h1'
  | 'h2'
  | 'h3'
  | 'h4'
  | 'h5'
  | 'h6'
  | 'subtitle1'
  | 'subtitle2'
  | 'body1'
  | 'body2'
  | 'caption'
  | 'button'
  | 'overline'
  | 'srOnly'
  | 'alignLeft'
  | 'alignCenter'
  | 'alignRight'
  | 'alignJustify'
  | 'noWrap'
  | 'gutterBottom'
  | 'paragraph'
  | 'colorInherit'
  | 'colorSecondary'
  | 'colorTextSecondary'
  | 'colorError'
  | 'displayInline'
  | 'displayBlock';
//export default Typography;
}
declare namespace MaterialUI {
export interface CardHeaderProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, CardHeaderClassKey, 'title'> {
  action?: React.ReactNode;
  avatar?: React.ReactNode;
  component?: React.ElementType<React.HTMLAttributes<HTMLDivElement>>;
  disableTypography?: boolean;
  subheader?: React.ReactNode;
  subheaderTypographyProps?: Partial<TypographyProps>;
  title?: React.ReactNode;
  titleTypographyProps?: Partial<TypographyProps>;
}
export type CardHeaderClassKey = 'root' | 'avatar' | 'action' | 'content' | 'title' | 'subheader';
export const CardHeader: React.ComponentType<CardHeaderProps>;
//export default CardHeader;
}
declare namespace MaterialUI {
export interface CardMediaTypeMap<P, D extends React.ElementType> {
  props: P & {
    image?: string;
    src?: string;
  };
  defaultComponent: D;
  classKey: CardMediaClassKey;
}
export const CardMedia: OverridableComponent<CardMediaTypeMap<{}, 'div'>>;
export type CardMediaClassKey = 'root' | 'media';
export type CardMediaProps<D extends React.ElementType = 'div', P = {}> = OverrideProps<
  CardMediaTypeMap<P, D>,
  D
>;
//export default CardMedia;
}
declare namespace MaterialUI {
export type IconButtonTypeMap<
  P = {},
  D extends React.ElementType = 'button'
> = ExtendButtonBaseTypeMap<{
  props: P & {
    color?: PropTypes.Color;
    disableFocusRipple?: boolean;
    edge?: 'start' | 'end' | false;
    size?: 'small' | 'medium';
  };
  defaultComponent: D;
  classKey: IconButtonClassKey;
}>;
export const IconButton: ExtendButtonBase<IconButtonTypeMap>;
export type IconButtonClassKey =
  | 'root'
  | 'edgeStart'
  | 'edgeEnd'
  | 'colorInherit'
  | 'colorPrimary'
  | 'colorSecondary'
  | 'disabled'
  | 'sizeSmall'
  | 'label';
export type IconButtonProps<
  D extends React.ElementType = IconButtonTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<IconButtonTypeMap<P, D>, D>;
//export default IconButton;
}
declare namespace MaterialUI {
export interface SwitchBaseProps
  extends StandardProps<IconButtonProps, SwitchBaseClassKey, 'onChange' | 'value'> {
  autoFocus?: boolean;
  checked?: boolean;
  checkedIcon: React.ReactNode;
  defaultChecked?: boolean;
  disabled?: boolean;
  disableRipple?: boolean;
  icon: React.ReactNode;
  inputProps?: React.InputHTMLAttributes<HTMLInputElement>;
  inputRef?: React.Ref<any>;
  name?: string;
  onChange?: (event: React.ChangeEvent<HTMLInputElement>, checked: boolean) => void;
  readOnly?: boolean;
  required?: boolean;
  tabIndex?: number;
  value?: unknown;
}
export type SwitchBaseClassKey = 'root' | 'checked' | 'disabled' | 'input';
export const SwitchBase: React.ComponentType<SwitchBaseProps>;
//export default SwitchBase;
}
declare namespace MaterialUI {
export interface CheckboxProps
  extends StandardProps<SwitchBaseProps, CheckboxClassKey, 'checkedIcon' | 'color' | 'icon'> {
  checkedIcon?: React.ReactNode;
  color?: 'primary' | 'secondary' | 'default';
  icon?: React.ReactNode;
  indeterminate?: boolean;
  indeterminateIcon?: React.ReactNode;
  size?: 'small' | 'medium';
}
export type CheckboxClassKey =
  | SwitchBaseClassKey
  | 'indeterminate'
  | 'colorPrimary'
  | 'colorSecondary';
export const Checkbox: React.ComponentType<CheckboxProps>;
//export default Checkbox;
}
declare namespace MaterialUI {
export interface ChipTypeMap<P = {}, D extends React.ElementType = 'div'> {
  props: P & {
    avatar?: React.ReactElement;
    clickable?: boolean;
    color?: PropTypes.Color;
    deleteIcon?: React.ReactElement;
    disabled?: boolean;
    icon?: React.ReactElement;
    label?: React.ReactNode;
    onDelete?: React.EventHandler<any>;
    size?: 'small' | 'medium';
    variant?: 'default' | 'outlined';
  };
  defaultComponent: D;
  classKey: ChipClassKey;
}
export const Chip: OverridableComponent<ChipTypeMap>;
export type ChipClassKey =
  | 'root'
  | 'sizeSmall'
  | 'colorPrimary'
  | 'colorSecondary'
  | 'disabled'
  | 'clickable'
  | 'clickableColorPrimary'
  | 'clickableColorSecondary'
  | 'deletable'
  | 'deletableColorPrimary'
  | 'deletableColorSecondary'
  | 'outlined'
  | 'outlinedPrimary'
  | 'outlinedSecondary'
  | 'avatar'
  | 'avatarSmall'
  | 'avatarColorPrimary'
  | 'avatarColorSecondary'
  | 'icon'
  | 'iconSmall'
  | 'iconColorPrimary'
  | 'iconColorSecondary'
  | 'label'
  | 'labelSmall'
  | 'deleteIcon'
  | 'deleteIconSmall'
  | 'deleteIconColorPrimary'
  | 'deleteIconColorSecondary'
  | 'deleteIconOutlinedColorPrimary'
  | 'deleteIconOutlinedColorSecondary';
export type ChipProps<
  D extends React.ElementType = ChipTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<ChipTypeMap<P, D>, D>;
//export default Chip;
}
declare namespace MaterialUI {
export interface CircularProgressProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, CircularProgressClassKey> {
  color?: 'primary' | 'secondary' | 'inherit';
  disableShrink?: boolean;
  size?: number | string;
  thickness?: number;
  value?: number;
  variant?: 'determinate' | 'indeterminate' | 'static';
}
export type CircularProgressClassKey =
  | 'root'
  | 'static'
  | 'indeterminate'
  | 'colorPrimary'
  | 'colorSecondary'
  | 'svg'
  | 'circle'
  | 'circleStatic'
  | 'circleIndeterminate'
  | 'circleDisableShrink';
export const CircularProgress: React.ComponentType<CircularProgressProps>;
//export default CircularProgress;
}
declare namespace MaterialUI {
export type TransitionHandlerKeys =
  | 'onEnter'
  | 'onEntering'
  | 'onEntered'
  | 'onExit'
  | 'onExiting'
  | 'onExited';
export type TransitionHandlerProps = Pick<_TransitionProps, TransitionHandlerKeys>;
export type TransitionKeys =
  | 'in'
  | 'mountOnEnter'
  | 'unmountOnExit'
  | 'timeout'
  | 'addEndListener'
  | TransitionHandlerKeys;
export interface TransitionProps
  extends TransitionActions,
    Partial<Pick<_TransitionProps, TransitionKeys>> {
  style?: CSSProperties;
}
}
declare namespace MaterialUI {
export interface CollapseProps extends StandardProps<TransitionProps, CollapseClassKey, 'timeout'> {
  children?: React.ReactNode;
  collapsedHeight?: string | number;
  component?: React.ElementType<TransitionProps>;
  theme?: Theme;
  timeout?: TransitionProps['timeout'] | 'auto';
}
export type CollapseClassKey = 'container' | 'entered' | 'hidden' | 'wrapper' | 'wrapperInner';
export const Collapse: React.ComponentType<CollapseProps>;
//export default Collapse;
}
declare namespace MaterialUI {
export interface ContainerProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, ContainerClassKey> {
  component?: React.ElementType<React.HTMLAttributes<HTMLDivElement>>;
  disableGutters?: boolean;
  fixed?: boolean;
  maxWidth?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | false;
}
export type ContainerClassKey =
  | 'root'
  | 'disableGutters'
  | 'fixed'
  | 'maxWidthXs'
  | 'maxWidthSm'
  | 'maxWidthMd'
  | 'maxWidthLg'
  | 'maxWidthXl';
export const Container: React.ComponentType<ContainerProps>;
//export default Container;
}
declare namespace MaterialUI {
export interface CssBaselineProps {
  children?: React.ReactElement;
}
export const CssBaseline: React.ComponentType<CssBaselineProps>;
export type CssBaselineClassKey = '@global';
//export default CssBaseline;
}
declare namespace MaterialUI {
export interface DialogActionsProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, DialogActionsClassKey> {
  disableSpacing?: boolean;
}
export type DialogActionsClassKey = 'root' | 'spacing';
export const DialogActions: React.ComponentType<DialogActionsProps>;
//export default DialogActions;
}
declare namespace MaterialUI {
export interface PortalProps {
  /**
   * The children to render into the `container`.
   */
  children?: React.ReactNode;
  /**
   * A node, component instance, or function that returns either.
   * The `container` will have the portal children appended to it.
   * By default, it uses the body of the top-level document object,
   * so it's simply `document.body` most of the time.
   */
  container?: React.ReactInstance | (() => React.ReactInstance | null) | null;
  /**
   * Disable the portal behavior.
   * The children stay within it's parent DOM hierarchy.
   */
  disablePortal?: boolean;
  /**
   * Callback fired once the children has been mounted into the `container`.
   *
   * This prop will be deprecated and removed in v5, the ref can be used instead.
   */
  onRendered?: () => void;
}
export function Portal(props: PortalProps): JSX.Element;
}
declare namespace MaterialUI {
export interface ModalProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, never, 'children'> {
  BackdropComponent?: React.ElementType<BackdropProps>;
  BackdropProps?: Partial<BackdropProps>;
  children?: React.ReactElement;
  closeAfterTransition?: boolean;
  container?: PortalProps['container'];
  disableAutoFocus?: boolean;
  disableBackdropClick?: boolean;
  disableEnforceFocus?: boolean;
  disableEscapeKeyDown?: boolean;
  disablePortal?: PortalProps['disablePortal'];
  disableRestoreFocus?: boolean;
  disableScrollLock?: boolean;
  hideBackdrop?: boolean;
  keepMounted?: boolean;
  manager?: ModalManager;
  onBackdropClick?: React.ReactEventHandler<{}>;
  onClose?: {
    bivarianceHack(event: {}, reason: 'backdropClick' | 'escapeKeyDown'): void;
  }['bivarianceHack'];
  onEscapeKeyDown?: React.ReactEventHandler<{}>;
  onRendered?: PortalProps['onRendered'];
  open: boolean;
}
export const Modal: React.ComponentType<ModalProps>;
//export default Modal;
}
declare namespace MaterialUI {
export interface DialogProps
  extends StandardProps<ModalProps & Partial<TransitionHandlerProps>, DialogClassKey, 'children'> {
  children?: React.ReactNode;
  fullScreen?: boolean;
  fullWidth?: boolean;
  maxWidth?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | false;
  PaperComponent?: React.ComponentType<PaperProps>;
  PaperProps?: Partial<PaperProps>;
  scroll?: 'body' | 'paper';
  TransitionComponent?: React.ComponentType<TransitionProps>;
  transitionDuration?: TransitionProps['timeout'];
  TransitionProps?: TransitionProps;
}
export type DialogClassKey =
  | 'root'
  | 'scrollPaper'
  | 'scrollBody'
  | 'container'
  | 'paper'
  | 'paperScrollPaper'
  | 'paperScrollBody'
  | 'paperWidthFalse'
  | 'paperWidthXs'
  | 'paperWidthSm'
  | 'paperWidthMd'
  | 'paperWidthLg'
  | 'paperWidthXl'
  | 'paperFullWidth'
  | 'paperFullScreen';
export const Dialog: React.ComponentType<DialogProps>;
//export default Dialog;
}
declare namespace MaterialUI {
export interface DialogContentProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, DialogContentClassKey> {
  dividers?: boolean;
}
export type DialogContentClassKey = 'root' | 'dividers';
export const DialogContent: React.ComponentType<DialogContentProps>;
//export default DialogContent;
}
declare namespace MaterialUI {
export interface DialogContentTextTypeMap<
  P = {},
  D extends React.ElementType = TypographyTypeMap['defaultComponent']
> {
  props: P & TypographyTypeMap['props'];
  defaultComponent: D;
  classKey: DialogContentTextClassKey;
}
export type DialogContentTextClassKey = 'root';
export const DialogContentText: OverridableComponent<DialogContentTextTypeMap>;
export type DialogContentTextProps<
  D extends React.ElementType = DialogContentTextTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<DialogContentTextTypeMap<P, D>, D>;
//export default DialogContentText;
}
declare namespace MaterialUI {
export interface DialogTitleProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, DialogTitleClassKey> {
  disableTypography?: boolean;
}
export type DialogTitleClassKey = 'root';
export const DialogTitle: React.ComponentType<DialogTitleProps>;
//export default DialogTitle;
}
declare namespace MaterialUI {
export interface DividerTypeMap<P = {}, D extends React.ElementType = 'hr'> {
  props: P & {
    absolute?: boolean;
    light?: boolean;
    orientation?: 'horizontal' | 'vertical';
    variant?: 'fullWidth' | 'inset' | 'middle';
  };
  defaultComponent: D;
  classKey: DividerClassKey;
}
export const Divider: OverridableComponent<DividerTypeMap>;
export type DividerClassKey = 'root' | 'absolute' | 'inset' | 'light' | 'middle' | 'vertical';
export type DividerProps<
  D extends React.ElementType = DividerTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<DividerTypeMap<P, D>, D>;
//export default Divider;
}
declare namespace MaterialUI {
export interface SlideProps extends TransitionProps {
  direction: 'left' | 'right' | 'up' | 'down';
  ref?: React.Ref<unknown>;
  theme?: Theme;
}
export const Slide: React.ComponentType<SlideProps>;
//export default Slide;
}
declare namespace MaterialUI {
export interface DrawerProps
  extends StandardProps<
    ModalProps & Partial<TransitionHandlerProps>,
    DrawerClassKey,
    'open' | 'children'
  > {
  anchor?: 'left' | 'top' | 'right' | 'bottom';
  children?: React.ReactNode;
  elevation?: number;
  ModalProps?: Partial<ModalProps>;
  open?: boolean;
  PaperProps?: Partial<PaperProps>;
  SlideProps?: Partial<SlideProps>;
  theme?: Theme;
  transitionDuration?: TransitionProps['timeout'];
  variant?: 'permanent' | 'persistent' | 'temporary';
}
export type DrawerClassKey =
  | 'root'
  | 'docked'
  | 'paper'
  | 'paperAnchorLeft'
  | 'paperAnchorRight'
  | 'paperAnchorTop'
  | 'paperAnchorBottom'
  | 'paperAnchorDockedLeft'
  | 'paperAnchorDockedTop'
  | 'paperAnchorDockedRight'
  | 'paperAnchorDockedBottom'
  | 'modal';
export const Drawer: React.ComponentType<DrawerProps>;
//export default Drawer;
}
declare namespace MaterialUI {
export interface ExpansionPanelActionsProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, ExpansionPanelActionsClassKey> {}
export type ExpansionPanelActionsClassKey = 'root' | 'spacing';
export const ExpansionPanelActions: React.ComponentType<ExpansionPanelActionsProps>;
//export default ExpansionPanelActions;
}
declare namespace MaterialUI {
export interface ExpansionPanelProps
  extends StandardProps<PaperProps, ExpansionPanelClassKey, 'onChange'> {
  defaultExpanded?: boolean;
  disabled?: boolean;
  expanded?: boolean;
  onChange?: (event: React.ChangeEvent<{}>, expanded: boolean) => void;
  TransitionComponent?: React.ComponentType<TransitionProps>;
  TransitionProps?: TransitionProps;
}
export type ExpansionPanelClassKey = 'root' | 'rounded' | 'expanded' | 'disabled';
export const ExpansionPanel: React.ComponentType<ExpansionPanelProps>;
//export default ExpansionPanel;
}
declare namespace MaterialUI {
export interface ExpansionPanelDetailsProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, ExpansionPanelDetailsClassKey> {}
export type ExpansionPanelDetailsClassKey = 'root';
export const ExpansionPanelDetails: React.ComponentType<ExpansionPanelDetailsProps>;
//export default ExpansionPanelDetails;
}
declare namespace MaterialUI {
export type ExpansionPanelSummaryTypeMap<
  P = {},
  D extends React.ElementType = 'div'
> = ExtendButtonBaseTypeMap<{
  props: P & {
    expandIcon?: React.ReactNode;
    IconButtonProps?: Partial<IconButtonProps>;
  };
  defaultComponent: D;
  classKey: ExpansionPanelSummaryClassKey;
}>;
export const ExpansionPanelSummary: ExtendButtonBase<ExpansionPanelSummaryTypeMap>;
export type ExpansionPanelSummaryClassKey =
  | 'root'
  | 'expanded'
  | 'focused'
  | 'disabled'
  | 'content'
  | 'expandIcon';
export type ExpansionPanelSummaryProps<
  D extends React.ElementType = ExpansionPanelSummaryTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<ExpansionPanelSummaryTypeMap<P, D>, D>;
//export default ExpansionPanelSummary;
}
declare namespace MaterialUI {
export type FabTypeMap<P = {}, D extends React.ElementType = 'button'> = ExtendButtonBaseTypeMap<{
  props: P & {
    color?: PropTypes.Color;
    disableFocusRipple?: boolean;
    href?: string;
    size?: 'small' | 'medium' | 'large';
    variant?: 'round' | 'extended';
  };
  defaultComponent: D;
  classKey: FabClassKey;
}>;
export const Fab: ExtendButtonBase<FabTypeMap>;
export type FabProps<
  D extends React.ElementType = FabTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<FabTypeMap<P, D>, D>;
export type FabClassKey =
  | 'root'
  | 'label'
  | 'primary'
  | 'secondary'
  | 'extended'
  | 'focusVisible'
  | 'disabled'
  | 'colorInherit'
  | 'sizeSmall'
  | 'sizeMedium';
//export default Fab;
}
declare namespace MaterialUI {
export interface InputBaseProps
  extends StandardProps<
    React.HTMLAttributes<HTMLDivElement>,
    InputBaseClassKey,
    'onChange' | 'onKeyUp' | 'onKeyDown' | 'onBlur' | 'onFocus' | 'defaultValue'
  > {
  autoComplete?: string;
  autoFocus?: boolean;
  color?: 'primary' | 'secondary';
  defaultValue?: unknown;
  disabled?: boolean;
  endAdornment?: React.ReactNode;
  error?: boolean;
  fullWidth?: boolean;
  id?: string;
  inputComponent?: React.ElementType<InputBaseComponentProps>;
  inputProps?: InputBaseComponentProps;
  inputRef?: React.Ref<any>;
  margin?: 'dense' | 'none';
  multiline?: boolean;
  name?: string;
  placeholder?: string;
  readOnly?: boolean;
  required?: boolean;
  renderPrefix?: (state: {
    disabled?: boolean;
    error?: boolean;
    filled?: boolean;
    focused?: boolean;
    margin?: 'dense' | 'none' | 'normal';
    required?: boolean;
    startAdornment?: React.ReactNode;
  }) => React.ReactNode;
  rows?: string | number;
  rowsMax?: string | number;
  rowsMin?: string | number;
  startAdornment?: React.ReactNode;
  type?: string;
  value?: unknown;
  /**
   * `onChange`, `onKeyUp`, `onKeyDown`, `onBlur`, `onFocus` are applied to the inner `InputComponent`,
   * which by default is an input or textarea. Since these handlers differ from the
   * ones inherited by `React.HTMLAttributes<HTMLDivElement>` we need to omit them.
   */
  onChange?: React.ChangeEventHandler<HTMLTextAreaElement | HTMLInputElement>;
  onKeyDown?: React.KeyboardEventHandler<HTMLTextAreaElement | HTMLInputElement>;
  onKeyUp?: React.KeyboardEventHandler<HTMLTextAreaElement | HTMLInputElement>;
  onBlur?: React.FocusEventHandler<HTMLInputElement | HTMLTextAreaElement>;
  onFocus?: React.FocusEventHandler<HTMLInputElement | HTMLTextAreaElement>;
}
export interface InputBaseComponentProps
  extends React.HTMLAttributes<HTMLInputElement | HTMLTextAreaElement> {
  // Accommodate arbitrary additional props coming from the `inputProps` prop
  [arbitrary: string]: any;
}
export type InputBaseClassKey =
  | 'root'
  | 'formControl'
  | 'focused'
  | 'disabled'
  | 'adornedEnd'
  | 'adornedStart'
  | 'error'
  | 'marginDense'
  | 'multiline'
  | 'fullWidth'
  | 'colorSecondary'
  | 'input'
  | 'inputMarginDense'
  | 'inputMultiline'
  | 'inputTypeSearch'
  | 'inputAdornedStart'
  | 'inputAdornedEnd'
  | 'inputHiddenLabel';
export const InputBase: React.ComponentType<InputBaseProps>;
//export default InputBase;
}
declare namespace MaterialUI {
export interface FilledInputProps extends StandardProps<InputBaseProps, FilledInputClassKey> {
  disableUnderline?: boolean;
}
export type FilledInputClassKey = InputBaseClassKey | 'colorSecondary' | 'underline';
export const FilledInput: React.ComponentType<FilledInputProps>;
//export default FilledInput;
}
declare namespace MaterialUI {
export interface FormControlTypeMap<P = {}, D extends React.ElementType = 'div'> {
  props: P & {
    color?: 'primary' | 'secondary';
    disabled?: boolean;
    error?: boolean;
    fullWidth?: boolean;
    hiddenLabel?: boolean;
    margin?: PropTypes.Margin;
    required?: boolean;
    size?: 'small' | 'medium';
    variant?: 'standard' | 'outlined' | 'filled';
  };
  defaultComponent: D;
  classKey: FormControlClassKey;
}
export const FormControl: OverridableComponent<FormControlTypeMap>;
export type FormControlClassKey = 'root' | 'marginNormal' | 'marginDense' | 'fullWidth';
export type FormControlProps<
  D extends React.ElementType = FormControlTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<FormControlTypeMap<P, D>, D>;
//export default FormControl;
}
declare namespace MaterialUI {
export interface FormControlLabelProps
  extends StandardProps<
    React.LabelHTMLAttributes<HTMLLabelElement>,
    FormControlLabelClassKey,
    'onChange'
  > {
  checked?: boolean;
  control: React.ReactElement;
  disabled?: boolean;
  inputRef?: React.Ref<any>;
  label: React.ReactNode;
  name?: string;
  onChange?: (event: React.ChangeEvent<{}>, checked: boolean) => void;
  labelPlacement?: 'end' | 'start' | 'top' | 'bottom';
  value?: unknown;
}
export type FormControlLabelClassKey =
  | 'root'
  | 'labelPlacementStart'
  | 'labelPlacementTop'
  | 'labelPlacementBottom'
  | 'disabled'
  | 'label';
export const FormControlLabel: React.ComponentType<FormControlLabelProps>;
//export default FormControlLabel;
}
declare namespace MaterialUI {
export interface FormGroupProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, FormGroupClassKey> {
  row?: boolean;
}
export type FormGroupClassKey = 'root' | 'row';
export const FormGroup: React.ComponentType<FormGroupProps>;
//export default FormGroup;
}
declare namespace MaterialUI {
export interface FormHelperTextProps
  extends StandardProps<React.HTMLAttributes<HTMLParagraphElement>, FormHelperTextClassKey> {
  disabled?: boolean;
  error?: boolean;
  filled?: boolean;
  focused?: boolean;
  component?: React.ElementType<React.HTMLAttributes<HTMLParagraphElement>>;
  margin?: 'dense';
  required?: boolean;
  variant?: 'standard' | 'outlined' | 'filled';
}
export type FormHelperTextClassKey =
  | 'root'
  | 'error'
  | 'disabled'
  | 'marginDense'
  | 'focused'
  | 'filled'
  | 'contained'
  | 'required';
export const FormHelperText: React.ComponentType<FormHelperTextProps>;
//export default FormHelperText;
}
declare namespace MaterialUI {
export interface FormLabelTypeMap<P = {}, D extends React.ElementType = 'label'> {
  props: P &
    FormLabelBaseProps & {
      color?: 'primary' | 'secondary';
      disabled?: boolean;
      error?: boolean;
      filled?: boolean;
      focused?: boolean;
      required?: boolean;
    };
  defaultComponent: D;
  classKey: FormLabelClassKey;
}
export const FormLabel: OverridableComponent<FormLabelTypeMap>;
export type FormLabelClassKey =
  | 'root'
  | 'colorSecondary'
  | 'focused'
  | 'disabled'
  | 'error'
  | 'filled'
  | 'required'
  | 'asterisk';
export type FormLabelBaseProps = React.LabelHTMLAttributes<HTMLLabelElement>;
export type FormLabelProps<
  D extends React.ElementType = FormLabelTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<FormLabelTypeMap<P, D>, D>;
//export default FormLabel;
}
declare namespace MaterialUI {
export type GridItemsAlignment = 'flex-start' | 'center' | 'flex-end' | 'stretch' | 'baseline';
export type GridContentAlignment =
  | 'stretch'
  | 'center'
  | 'flex-start'
  | 'flex-end'
  | 'space-between'
  | 'space-around';
export type GridDirection = 'row' | 'row-reverse' | 'column' | 'column-reverse';
export type GridSpacing = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10;
export type GridJustification =
  | 'flex-start'
  | 'center'
  | 'flex-end'
  | 'space-between'
  | 'space-around'
  | 'space-evenly';
export type GridWrap = 'nowrap' | 'wrap' | 'wrap-reverse';
export type GridSize = 'auto' | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;
export type GridClassKey =
  | 'root'
  | 'container'
  | 'item'
  | 'zeroMinWidth'
  | 'direction-xs-column'
  | 'direction-xs-column-reverse'
  | 'direction-xs-row-reverse'
  | 'wrap-xs-nowrap'
  | 'wrap-xs-wrap-reverse'
  | 'align-items-xs-center'
  | 'align-items-xs-flex-start'
  | 'align-items-xs-flex-end'
  | 'align-items-xs-baseline'
  | 'align-content-xs-center'
  | 'align-content-xs-flex-start'
  | 'align-content-xs-flex-end'
  | 'align-content-xs-space-between'
  | 'align-content-xs-space-around'
  | 'justify-xs-center'
  | 'justify-xs-flex-end'
  | 'justify-xs-space-between'
  | 'justify-xs-space-around'
  | 'justify-xs-space-evenly'
  | 'spacing-xs-1'
  | 'spacing-xs-2'
  | 'spacing-xs-3'
  | 'spacing-xs-4'
  | 'spacing-xs-5'
  | 'spacing-xs-6'
  | 'spacing-xs-7'
  | 'spacing-xs-8'
  | 'spacing-xs-9'
  | 'spacing-xs-10'
  | 'grid-xs-auto'
  | 'grid-xs-true'
  | 'grid-xs-1'
  | 'grid-xs-2'
  | 'grid-xs-3'
  | 'grid-xs-4'
  | 'grid-xs-5'
  | 'grid-xs-6'
  | 'grid-xs-7'
  | 'grid-xs-8'
  | 'grid-xs-9'
  | 'grid-xs-10'
  | 'grid-xs-11'
  | 'grid-xs-12';
export interface GridTypeMap<P = {}, D extends React.ElementType = 'div'> {
  props: P &
    Partial<Record<Breakpoint, boolean | GridSize>> & {
      alignContent?: GridContentAlignment;
      alignItems?: GridItemsAlignment;
      container?: boolean;
      direction?: GridDirection;
      item?: boolean;
      justify?: GridJustification;
      spacing?: GridSpacing;
      wrap?: GridWrap;
      zeroMinWidth?: boolean;
    };
  defaultComponent: D;
  classKey: GridClassKey;
}
export const Grid: OverridableComponent<GridTypeMap>;
export type GridProps<
  D extends React.ElementType = GridTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<GridTypeMap<P, D>, D>;
//export default Grid;
}
declare namespace MaterialUI {
export interface GridListProps
  extends StandardProps<React.HTMLAttributes<HTMLUListElement>, GridListClassKey> {
  cellHeight?: number | 'auto';
  cols?: number;
  component?: React.ElementType<React.HTMLAttributes<HTMLUListElement>>;
  spacing?: number;
}
export type GridListClassKey = 'root';
export const GridList: React.ComponentType<GridListProps>;
//export default GridList;
}
declare namespace MaterialUI {
export interface GridListTileBarProps extends StandardProps<{}, GridListTileBarClassKey> {
  actionIcon?: React.ReactNode;
  actionPosition?: 'left' | 'right';
  subtitle?: React.ReactNode;
  title?: React.ReactNode;
  titlePosition?: 'top' | 'bottom';
}
export type GridListTileBarClassKey =
  | 'root'
  | 'titlePositionBottom'
  | 'titlePositionTop'
  | 'rootSubtitle'
  | 'titleWrap'
  | 'titleWrapActionPosLeft'
  | 'titleWrapActionPosRight'
  | 'title'
  | 'subtitle'
  | 'actionIcon'
  | 'actionIconActionPosLeft';
export const GridListTileBar: React.ComponentType<GridListTileBarProps>;
//export default GridListTileBar;
}
declare namespace MaterialUI {
export interface GridListTileProps
  extends StandardProps<React.HTMLAttributes<HTMLLIElement>, GridListTileClassKey> {
  cols?: number;
  component?: React.ElementType<React.HTMLAttributes<HTMLLIElement>>;
  rows?: number;
}
export type GridListTileClassKey = 'root' | 'tile' | 'imgFullHeight' | 'imgFullWidth';
export const GridListTile: React.ComponentType<GridListTileProps>;
//export default GridListTile;
}
declare namespace MaterialUI {
export interface IconProps
  extends StandardProps<React.HTMLAttributes<HTMLSpanElement>, IconClassKey> {
  color?: PropTypes.Color | 'action' | 'disabled' | 'error';
  component?: React.ElementType<React.HTMLAttributes<HTMLSpanElement>>;
  fontSize?: 'inherit' | 'default' | 'small' | 'large';
}
export type IconClassKey =
  | 'root'
  | 'colorSecondary'
  | 'colorAction'
  | 'colorDisabled'
  | 'colorError'
  | 'colorPrimary'
  | 'fontSizeInherit'
  | 'fontSizeSmall'
  | 'fontSizeLarge';
export const Icon: React.ComponentType<IconProps>;
//export default Icon;
}
declare namespace MaterialUI {
export interface InputAdornmentProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, InputAdornmentClassKey> {
  component?: React.ElementType<React.HTMLAttributes<HTMLDivElement>>;
  disablePointerEvents?: boolean;
  disableTypography?: boolean;
  position: 'start' | 'end';
  variant?: 'standard' | 'outlined' | 'filled';
}
export type InputAdornmentClassKey =
  | 'root'
  | 'filled'
  | 'positionStart'
  | 'positionEnd'
  | 'disablePointerEvents'
  | 'hiddenLabel'
  | 'marginDense';
export const InputAdornment: React.ComponentType<InputAdornmentProps>;
//export default InputAdornment;
}
declare namespace MaterialUI {
export interface InputProps extends StandardProps<InputBaseProps, InputClassKey> {
  disableUnderline?: boolean;
}
export type InputClassKey =
  | 'root'
  | 'formControl'
  | 'focused'
  | 'disabled'
  | 'colorSecondary'
  | 'underline'
  | 'error'
  | 'marginDense'
  | 'multiline'
  | 'fullWidth'
  | 'input'
  | 'inputMarginDense'
  | 'inputMultiline'
  | 'inputTypeSearch';
export const Input: React.ComponentType<InputProps>;
//export default Input;
}
declare namespace MaterialUI {
export interface InputLabelProps extends StandardProps<FormLabelProps, InputLabelClassKey> {
  disableAnimation?: boolean;
  disabled?: boolean;
  error?: boolean;
  focused?: boolean;
  margin?: 'dense';
  required?: boolean;
  shrink?: boolean;
  variant?: 'standard' | 'outlined' | 'filled';
}
export type InputLabelClassKey =
  | 'root'
  | 'focused'
  | 'disabled'
  | 'error'
  | 'required'
  | 'asterisk'
  | 'formControl'
  | 'marginDense'
  | 'shrink'
  | 'animated'
  | 'filled'
  | 'outlined';
export const InputLabel: React.ComponentType<InputLabelProps>;
//export default InputLabel;
}
declare namespace MaterialUI {
export interface LinearProgressProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, LinearProgressClassKey> {
  color?: 'primary' | 'secondary';
  value?: number;
  valueBuffer?: number;
  variant?: 'determinate' | 'indeterminate' | 'buffer' | 'query';
}
export type LinearProgressClassKey =
  | 'root'
  | 'colorPrimary'
  | 'colorSecondary'
  | 'determinate'
  | 'indeterminate'
  | 'buffer'
  | 'query'
  | 'dashed'
  | 'dashedColorPrimary'
  | 'dashedColorSecondary'
  | 'bar'
  | 'barColorPrimary'
  | 'barColorSecondary'
  | 'bar1Indeterminate'
  | 'bar2Indeterminate'
  | 'bar1Determinate'
  | 'bar1Buffer'
  | 'bar2Buffer';
export const LinearProgress: React.ComponentType<LinearProgressProps>;
//export default LinearProgress;
}
declare namespace MaterialUI {
export interface LinkTypeMap<P = {}, D extends React.ElementType = 'a'> {
  props: P &
    LinkBaseProps & {
      TypographyClasses?: TypographyProps['classes'];
      underline?: 'none' | 'hover' | 'always';
    };
  defaultComponent: D;
  classKey: LinkClassKey;
}
export const Link: OverridableComponent<LinkTypeMap>;
export type LinkClassKey =
  | 'root'
  | 'underlineNone'
  | 'underlineHover'
  | 'underlineAlways'
  | 'button'
  | 'focusVisible';
export type LinkBaseProps = React.AnchorHTMLAttributes<HTMLAnchorElement> &
  Omit<TypographyProps, 'component'>;
export type LinkProps<
  D extends React.ElementType = LinkTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<LinkTypeMap<P, D>, D>;
//export default Link;
}
declare namespace MaterialUI {
export interface ListTypeMap<P = {}, D extends React.ElementType = 'ul'> {
  props: P & {
    dense?: boolean;
    disablePadding?: boolean;
    subheader?: React.ReactElement;
  };
  defaultComponent: D;
  classKey: ListClassKey;
}
export const List: OverridableComponent<ListTypeMap>;
export type ListClassKey = 'root' | 'padding' | 'dense' | 'subheader';
export type ListProps<
  D extends React.ElementType = ListTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<ListTypeMap<P, D>, D>;
//export default List;
}
declare namespace MaterialUI {
export interface ListItemAvatarProps extends StandardProps<{}, ListItemAvatarClassKey> {}
export type ListItemAvatarClassKey = 'root' | 'icon';
export const ListItemAvatar: React.ComponentType<ListItemAvatarProps>;
//export default ListItemAvatar;
}
declare namespace MaterialUI {
export interface ListItemTypeMap<P, D extends React.ElementType> {
  props: P & {
    alignItems?: 'flex-start' | 'center';
    autoFocus?: boolean;
    button?: boolean;
    ContainerComponent?: React.ElementType<React.HTMLAttributes<HTMLDivElement>>;
    ContainerProps?: React.HTMLAttributes<HTMLDivElement>;
    dense?: boolean;
    disabled?: boolean;
    disableGutters?: boolean;
    divider?: boolean;
    focusVisibleClassName?: string;
    selected?: boolean;
  };
  defaultComponent: D;
  classKey: ListItemClassKey;
}
export const ListItem: OverridableComponent<ListItemTypeMap<{ button?: false }, 'li'>> &
  ExtendButtonBase<ListItemTypeMap<{ button: true }, 'div'>>;
export type ListItemClassKey =
  | 'root'
  | 'container'
  | 'focusVisible'
  | 'default'
  | 'dense'
  | 'disabled'
  | 'divider'
  | 'gutters'
  | 'button'
  | 'secondaryAction'
  | 'selected';
export type ListItemProps<D extends React.ElementType = 'li', P = {}> = OverrideProps<
  ListItemTypeMap<P, D>,
  D
>;
//export default ListItem;
}
declare namespace MaterialUI {
export interface ListItemIconProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, ListItemIconClassKey> {
  children?: React.ReactElement;
}
export type ListItemIconClassKey = 'root';
export const ListItemIcon: React.ComponentType<ListItemIconProps>;
//export default ListItemIcon;
}
declare namespace MaterialUI {
export interface ListItemSecondaryActionProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, ListItemSecondaryActionClassKey> {}
export type ListItemSecondaryActionClassKey = 'root';
export const ListItemSecondaryAction: React.ComponentType<ListItemSecondaryActionProps>;
//export default ListItemSecondaryAction;
}
declare namespace MaterialUI {
export interface ListItemTextProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, ListItemTextClassKey> {
  disableTypography?: boolean;
  inset?: boolean;
  primary?: React.ReactNode;
  primaryTypographyProps?: Partial<TypographyProps>;
  secondary?: React.ReactNode;
  secondaryTypographyProps?: Partial<TypographyProps>;
}
export type ListItemTextClassKey =
  | 'root'
  | 'multiline'
  | 'dense'
  | 'inset'
  | 'primary'
  | 'secondary';
export const ListItemText: React.ComponentType<ListItemTextProps>;
//export default ListItemText;
}
declare namespace MaterialUI {
export interface ListSubheaderTypeMap<P = {}, D extends React.ElementType = 'li'> {
  props: P & {
    color?: 'default' | 'primary' | 'inherit';
    disableGutters?: boolean;
    disableSticky?: boolean;
    inset?: boolean;
  };
  defaultComponent: D;
  classKey: ListSubheaderClassKey;
}
export const ListSubheader: OverridableComponent<ListSubheaderTypeMap>;
export type ListSubheaderClassKey =
  | 'root'
  | 'colorPrimary'
  | 'colorInherit'
  | 'inset'
  | 'sticky'
  | 'gutters';
export type ListSubheaderProps<
  D extends React.ElementType = ListSubheaderTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<ListSubheaderTypeMap<P, D>, D>;
//export default ListSubheader;
}
declare namespace MaterialUI {
export interface PopoverOrigin {
  vertical: 'top' | 'center' | 'bottom' | number;
  horizontal: 'left' | 'center' | 'right' | number;
}
export interface PopoverPosition {
  top: number;
  left: number;
}
export type PopoverReference = 'anchorEl' | 'anchorPosition' | 'none';
export interface PopoverProps
  extends StandardProps<ModalProps & Partial<TransitionHandlerProps>, PopoverClassKey, 'children'> {
  action?: React.Ref<PopoverActions>;
  anchorEl?: null | Element | ((element: Element) => Element);
  anchorOrigin?: PopoverOrigin;
  anchorPosition?: PopoverPosition;
  anchorReference?: PopoverReference;
  children?: React.ReactNode;
  elevation?: number;
  getContentAnchorEl?: null | ((element: Element) => Element);
  marginThreshold?: number;
  modal?: boolean;
  PaperProps?: Partial<PaperProps>;
  role?: string;
  transformOrigin?: PopoverOrigin;
  TransitionComponent?: React.ComponentType<TransitionProps>;
  transitionDuration?: TransitionProps['timeout'] | 'auto';
  TransitionProps?: TransitionProps;
}
export type PopoverClassKey = 'root' | 'paper';
export interface PopoverActions {
  updatePosition(): void;
}
export const Popover: React.ComponentType<PopoverProps>;
//export default Popover;
}
declare namespace MaterialUI {
export interface MenuListProps extends StandardProps<ListProps, MenuListClassKey> {
  autoFocus?: boolean;
  autoFocusItem?: boolean;
  disableListWrap?: boolean;
  variant?: 'menu' | 'selectedMenu';
}
export type MenuListClassKey = ListClassKey;
export const MenuList: React.ComponentType<MenuListProps>;
//export default MenuList;
}
declare namespace MaterialUI {
export interface MenuProps
  extends StandardProps<PopoverProps & Partial<TransitionHandlerProps>, MenuClassKey> {
  autoFocus?: boolean;
  disableAutoFocusItem?: boolean;
  MenuListProps?: Partial<MenuListProps>;
  PaperProps?: Partial<PaperProps>;
  PopoverClasses?: PopoverProps['classes'];
  transitionDuration?: TransitionProps['timeout'] | 'auto';
  variant?: 'menu' | 'selectedMenu';
}
export type MenuClassKey = 'paper' | 'list';
export const Menu: React.ComponentType<MenuProps>;
//export default Menu;
}
declare namespace MaterialUI {
export type MenuItemClassKey = 'root' | 'gutters' | 'selected' | 'dense';
export type MenuItemTypeMap<P = {}, D extends React.ElementType = 'li'> = Omit<
  ListItemTypeMap<P, D>,
  'classKey'
> & {
  classKey: MenuItemClassKey;
};
export const MenuItem: OverridableComponent<
  MenuItemTypeMap<{ button: false }, MenuItemTypeMap['defaultComponent']>
> &
  ExtendButtonBase<MenuItemTypeMap<{ button?: true }, MenuItemTypeMap['defaultComponent']>>;
export type MenuItemProps<
  D extends React.ElementType = MenuItemTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<MenuItemTypeMap<P, D>, D>;
//export default MenuItem;
}
declare namespace MaterialUI {
export interface MobileStepperProps
  extends StandardProps<PaperProps, MobileStepperClassKey, 'variant'> {
  activeStep?: number;
  backButton: React.ReactElement;
  LinearProgressProps?: Partial<LinearProgressProps>;
  nextButton: React.ReactElement;
  position?: 'bottom' | 'top' | 'static';
  steps: number;
  variant?: 'text' | 'dots' | 'progress';
}
export type MobileStepperClassKey =
  | 'root'
  | 'positionBottom'
  | 'positionTop'
  | 'positionStatic'
  | 'dots'
  | 'dot'
  | 'dotActive'
  | 'progress';
export const MobileStepper: React.ComponentType<MobileStepperProps>;
//export default MobileStepper;
}
declare namespace MaterialUI {
export interface NativeSelectInputProps {
  disabled?: boolean;
  IconComponent?: React.ElementType;
  inputRef?: (
    ref: HTMLSelectElement | { node: HTMLInputElement; value: NativeSelectInputProps['value'] },
  ) => void;
  name?: string;
  onChange?: (event: React.ChangeEvent<HTMLSelectElement>, child: React.ReactNode) => void;
  value?: unknown;
  variant?: 'standard' | 'outlined' | 'filled';
}
export const NativeSelectInput: React.ComponentType<NativeSelectInputProps>;
//export default NativeSelectInput;
}
declare namespace MaterialUI {
export interface NativeSelectProps
  extends StandardProps<InputProps, NativeSelectClassKey, 'value' | 'onChange'>,
    Pick<NativeSelectInputProps, 'onChange'> {
  IconComponent?: React.ElementType;
  input?: React.ReactNode;
  value?: unknown;
  variant?: 'standard' | 'outlined' | 'filled';
}
export type NativeSelectClassKey =
  | 'root'
  | 'select'
  | 'filled'
  | 'outlined'
  | 'selectMenu'
  | 'disabled'
  | 'icon'
  | 'iconFilled'
  | 'iconOutlined';
export const NativeSelect: React.ComponentType<NativeSelectProps>;
//export default NativeSelect;
}
declare namespace MaterialUI {
export interface OutlinedInputProps extends StandardProps<InputBaseProps, OutlinedInputClassKey> {
  label?: React.ReactNode;
  notched?: boolean;
  labelWidth?: number;
}
export type OutlinedInputClassKey =
  | 'root'
  | 'colorSecondary'
  | 'focused'
  | 'disabled'
  | 'adornedStart'
  | 'adornedEnd'
  | 'error'
  | 'marginDense'
  | 'multiline'
  | 'notchedOutline'
  | 'input'
  | 'inputMarginDense'
  | 'inputMultiline'
  | 'inputAdornedStart'
  | 'inputAdornedEnd';
export const OutlinedInput: React.ComponentType<OutlinedInputProps>;
//export default OutlinedInput;
}
declare namespace MaterialUI {
export interface RadioProps
  extends StandardProps<SwitchBaseProps, RadioClassKey, 'checkedIcon' | 'color' | 'icon'> {
  checkedIcon?: React.ReactNode;
  color?: 'primary' | 'secondary' | 'default';
  icon?: React.ReactNode;
  size?: 'small' | 'medium';
}
export type RadioClassKey = SwitchBaseClassKey | 'colorPrimary' | 'colorSecondary';
export const Radio: React.ComponentType<RadioProps>;
//export default Radio;
}
declare namespace MaterialUI {
export interface SelectInputProps {
  autoFocus?: boolean;
  autoWidth: boolean;
  disabled?: boolean;
  IconComponent?: React.ElementType;
  inputRef?: (
    ref: HTMLSelectElement | { node: HTMLInputElement; value: SelectInputProps['value'] },
  ) => void;
  MenuProps?: Partial<MenuProps>;
  multiple: boolean;
  name?: string;
  native: boolean;
  onBlur?: React.FocusEventHandler<any>;
  onChange?: (
    event: React.ChangeEvent<{ name?: string; value: unknown }>,
    child: React.ReactNode,
  ) => void;
  onClose?: (event: React.ChangeEvent<{}>) => void;
  onFocus?: React.FocusEventHandler<any>;
  onOpen?: (event: React.ChangeEvent<{}>) => void;
  open?: boolean;
  readOnly?: boolean;
  renderValue?: (value: SelectInputProps['value']) => React.ReactNode;
  SelectDisplayProps?: React.HTMLAttributes<HTMLDivElement>;
  tabIndex?: number;
  value?: unknown;
  variant?: 'standard' | 'outlined' | 'filled';
}
export const SelectInput: React.ComponentType<SelectInputProps>;
//export default SelectInput;
}
declare namespace MaterialUI {
export interface SelectProps
  extends StandardProps<InputProps, SelectClassKey, 'value' | 'onChange'>,
    Pick<SelectInputProps, 'onChange'> {
  autoWidth?: boolean;
  displayEmpty?: boolean;
  IconComponent?: React.ElementType;
  input?: React.ReactNode;
  label?: React.ReactNode;
  labelId?: string;
  labelWidth?: number;
  MenuProps?: Partial<MenuProps>;
  multiple?: boolean;
  native?: boolean;
  onClose?: (event: React.ChangeEvent<{}>) => void;
  onOpen?: (event: React.ChangeEvent<{}>) => void;
  open?: boolean;
  renderValue?: (value: SelectProps['value']) => React.ReactNode;
  SelectDisplayProps?: React.HTMLAttributes<HTMLDivElement>;
  value?: unknown;
  variant?: 'standard' | 'outlined' | 'filled';
}
export type SelectClassKey =
  | 'root'
  | 'select'
  | 'filled'
  | 'outlined'
  | 'selectMenu'
  | 'disabled'
  | 'icon'
  | 'iconOpen'
  | 'iconFilled'
  | 'iconOutlined';
export const Select: React.ComponentType<SelectProps>;
//export default Select;
}
declare namespace MaterialUI {
export interface Mark {
  value: number;
  label?: React.ReactNode;
}
export interface ValueLabelProps extends React.HTMLAttributes<HTMLSpanElement> {
  value: number;
  open: boolean;
  children?: React.ReactElement;
}
export interface SliderProps
  extends StandardProps<
    React.HTMLAttributes<HTMLSpanElement>,
    SliderClassKey,
    'defaultValue' | 'onChange'
  > {
  'aria-label'?: string;
  'aria-labelledby'?: string;
  'aria-valuetext'?: string;
  color?: 'primary' | 'secondary';
  component?: React.ElementType<React.HTMLAttributes<HTMLSpanElement>>;
  defaultValue?: number | number[];
  disabled?: boolean;
  getAriaLabel?: (index: number) => string;
  getAriaValueText?: (value: number, index: number) => string;
  marks?: boolean | Mark[];
  max?: number;
  min?: number;
  name?: string;
  onChange?: (event: React.ChangeEvent<{}>, value: number | number[]) => void;
  onChangeCommitted?: (event: React.ChangeEvent<{}>, value: number | number[]) => void;
  orientation?: 'horizontal' | 'vertical';
  step?: number | null;
  scale?: (value: number) => number;
  ThumbComponent?: React.ElementType<React.HTMLAttributes<HTMLSpanElement>>;
  track?: 'normal' | false | 'inverted';
  value?: number | number[];
  ValueLabelComponent?: React.ElementType<ValueLabelProps>;
  valueLabelDisplay?: 'on' | 'auto' | 'off';
  valueLabelFormat?: string | ((value: number, index: number) => React.ReactNode);
}
export type SliderClassKey =
  | 'root'
  | 'colorPrimary'
  | 'colorSecondary'
  | 'marked'
  | 'vertical'
  | 'rtl'
  | 'disabled'
  | 'rail'
  | 'track'
  | 'trackFalse'
  | 'trackInverted'
  | 'thumb'
  | 'thumbColorPrimary'
  | 'thumbColorSecondary'
  | 'valueLabel'
  | 'mark'
  | 'markActive'
  | 'markLabel'
  | 'markLabelActive';
export const Slider: React.ComponentType<SliderProps>;
//export default Slider;
}
declare namespace MaterialUI {
export interface SnackbarContentProps extends StandardProps<PaperProps, SnackbarContentClassKey> {
  action?: React.ReactNode;
  message?: React.ReactNode;
}
export type SnackbarContentClassKey = 'root' | 'message' | 'action';
export const SnackbarContent: React.ComponentType<SnackbarContentProps>;
//export default SnackbarContent;
}
declare namespace MaterialUI {
export interface ClickAwayListenerProps {
  children?: React.ReactNode;
  mouseEvent?: 'onClick' | 'onMouseDown' | 'onMouseUp' | false;
  onClickAway: (event: React.MouseEvent<Document>) => void;
  touchEvent?: 'onTouchStart' | 'onTouchEnd' | false;
}
export const ClickAwayListener: React.ComponentType<ClickAwayListenerProps>;
//export default ClickAwayListener;
}
declare namespace MaterialUI {
export interface SnackbarOrigin {
  vertical: 'top' | 'bottom';
  horizontal: 'left' | 'center' | 'right';
}
export interface SnackbarProps
  extends StandardProps<
    React.HTMLAttributes<HTMLDivElement> & Partial<TransitionHandlerProps>,
    SnackbarClassKey
  > {
  action?: SnackbarContentProps['action'];
  anchorOrigin?: SnackbarOrigin;
  autoHideDuration?: number | null;
  ClickAwayListenerProps?: Partial<ClickAwayListenerProps>;
  ContentProps?: Partial<SnackbarContentProps>;
  disableWindowBlurListener?: boolean;
  message?: SnackbarContentProps['message'];
  onClose?: (event: React.SyntheticEvent<any>, reason: string) => void;
  onMouseEnter?: React.MouseEventHandler<any>;
  onMouseLeave?: React.MouseEventHandler<any>;
  open: boolean;
  resumeHideDuration?: number;
  TransitionComponent?: React.ComponentType<TransitionProps>;
  transitionDuration?: TransitionProps['timeout'];
  TransitionProps?: TransitionProps;
}
export type SnackbarClassKey =
  | 'root'
  | 'anchorOriginTopCenter'
  | 'anchorOriginBottomCenter'
  | 'anchorOriginTopRight'
  | 'anchorOriginBottomRight'
  | 'anchorOriginTopLeft'
  | 'anchorOriginBottomLeft';
export const Snackbar: React.ComponentType<SnackbarProps>;
//export default Snackbar;
}
declare namespace MaterialUI {
export type Orientation = 'horizontal' | 'vertical';
export interface StepperProps extends StandardProps<PaperProps, StepperClasskey> {
  activeStep?: number;
  alternativeLabel?: boolean;
  children?: React.ReactNode;
  connector?: React.ReactElement | React.ReactNode;
  nonLinear?: boolean;
  orientation?: Orientation;
}
export type StepperClasskey = 'root' | 'horizontal' | 'vertical' | 'alternativeLabel';
export const Stepper: React.ComponentType<StepperProps>;
//export default Stepper;
}
declare namespace MaterialUI {
export type StepButtonIcon = React.ReactElement | string | number | null;
export type StepButtonTypeMap<P, D extends React.ElementType> = ExtendButtonBaseTypeMap<{
  props: P & {
    active?: boolean;
    alternativeLabel?: boolean;
    completed?: boolean;
    disabled?: boolean;
    icon?: StepButtonIcon;
    last?: boolean;
    optional?: React.ReactNode;
    orientation?: Orientation;
  };
  defaultComponent: D;
  classKey: StepButtonClasskey;
}>;
export const StepButton: ExtendButtonBase<
  StepButtonTypeMap<{}, ButtonBaseTypeMap['defaultComponent']>
>;
export type StepButtonClasskey = 'root' | 'vertical' | 'touchRipple';
export type StepButtonProps<
  D extends React.ElementType = ButtonBaseTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<StepButtonTypeMap<P, D>, D>;
//export default StepButton;
}
declare namespace MaterialUI {
export interface StepProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, StepClasskey> {
  active?: boolean;
  alternativeLabel?: boolean;
  children?: React.ReactNode;
  completed?: boolean;
  connector?: React.ReactElement;
  disabled?: boolean;
  index?: number;
  last?: boolean;
  orientation?: Orientation;
}
export type StepClasskey = 'root' | 'horizontal' | 'vertical' | 'alternativeLabel' | 'completed';
export const Step: React.ComponentType<StepProps>;
//export default Step;
}
declare namespace MaterialUI {
export type StepConnectorIcon = React.ReactElement | string | number;
export interface StepConnectorProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, StepConnectorClasskey> {
  active?: boolean;
  alternativeLabel?: boolean;
  completed?: boolean;
  disabled?: boolean;
  index?: number;
  orientation?: Orientation;
}
export type StepConnectorClasskey =
  | 'root'
  | 'horizontal'
  | 'vertical'
  | 'alternativeLabel'
  | 'active'
  | 'completed'
  | 'disabled'
  | 'line'
  | 'lineHorizontal'
  | 'lineVertical';
export const StepConnector: React.ComponentType<StepConnectorProps>;
//export default StepConnector;
}
declare namespace MaterialUI {
export interface StepContentProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, StepContentClasskey> {
  active?: boolean;
  alternativeLabel?: boolean;
  children?: React.ReactNode;
  completed?: boolean;
  last?: boolean;
  optional?: boolean;
  orientation?: Orientation;
  TransitionComponent?: React.ComponentType<TransitionProps>;
  transitionDuration?: TransitionProps['timeout'] | 'auto';
  TransitionProps?: TransitionProps;
}
export type StepContentClasskey = 'root' | 'last' | 'transition';
export const StepContent: React.ComponentType<StepContentProps>;
//export default StepContent;
}
declare namespace MaterialUI {
export interface StepIconProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, StepIconClasskey> {
  active?: boolean;
  completed?: boolean;
  error?: boolean;
  icon: React.ReactNode;
}
export type StepIconClasskey = 'root' | 'text' | 'active' | 'completed' | 'error';
export const StepIcon: React.ComponentType<StepIconProps>;
//export default StepIcon;
}
declare namespace MaterialUI {
export interface StepLabelProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, StepLabelClasskey> {
  active?: boolean;
  alternativeLabel?: boolean;
  children?: React.ReactNode;
  completed?: boolean;
  disabled?: boolean;
  error?: boolean;
  icon?: StepButtonIcon;
  last?: boolean;
  optional?: React.ReactNode;
  orientation?: Orientation;
  StepIconComponent?: React.ElementType;
  StepIconProps?: Partial<StepIconProps>;
}
export type StepLabelClasskey =
  | 'root'
  | 'horizontal'
  | 'vertical'
  | 'active'
  | 'completed'
  | 'alternativeLabel'
  | 'error'
  | 'disabled'
  | 'label'
  | 'iconContainer'
  | 'labelContainer';
export const StepLabel: React.ComponentType<StepLabelProps>;
//export default StepLabel;
}
declare namespace MaterialUI {
export interface SvgIconProps
  extends StandardProps<React.SVGProps<SVGSVGElement>, SvgIconClassKey, 'children'> {
  /**
   * Node passed into the SVG element.
   */
  children?: React.ReactNode;
  /**
   * The color of the component. It supports those theme colors that make sense for this component.
   * You can use the `htmlColor` prop to apply a color attribute to the SVG element.
   */
  color?: 'inherit' | 'primary' | 'secondary' | 'action' | 'disabled' | 'error';
  /**
   * The component used for the root node.
   * Either a string to use a DOM element or a component.
   */
  component?: React.ElementType<React.SVGProps<SVGSVGElement>>;
  /**
   * The fontSize applied to the icon. Defaults to 24px, but can be configure to inherit font size.
   */
  fontSize?: 'inherit' | 'default' | 'small' | 'large';
  /**
   * Applies a color attribute to the SVG element.
   */
  htmlColor?: string;
  /**
   * The shape-rendering attribute. The behavior of the different options is described on the
   * [MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/shape-rendering).
   * If you are having issues with blurry icons you should investigate this property.
   * @document
   */
  shapeRendering?: string;
  /**
   * Provides a human-readable title for the element that contains it.
   * https://www.w3.org/TR/SVG-access/#Equivalent
   */
  titleAccess?: string;
  /**
   * Allows you to redefine what the coordinates without units mean inside an SVG element.
   * For example, if the SVG element is 500 (width) by 200 (height),
   * and you pass viewBox="0 0 50 20",
   * this means that the coordinates inside the SVG will go from the top left corner (0,0)
   * to bottom right (50,20) and each unit will be worth 10px.
   */
  viewBox?: string;
}
export type SvgIconClassKey =
  | 'root'
  | 'colorSecondary'
  | 'colorAction'
  | 'colorDisabled'
  | 'colorError'
  | 'colorPrimary'
  | 'fontSizeInherit'
  | 'fontSizeSmall'
  | 'fontSizeLarge';
export function SvgIcon(props: SvgIconProps): JSX.Element;
}
declare namespace MaterialUI {
export interface SwitchProps
  extends StandardProps<SwitchBaseProps, SwitchClassKey, 'checkedIcon' | 'color' | 'icon'> {
  checkedIcon?: React.ReactNode;
  color?: 'primary' | 'secondary' | 'default';
  icon?: React.ReactNode;
  size?: 'small' | 'medium';
}
export type SwitchClassKey =
  | SwitchBaseClassKey
  | 'switchBase'
  | 'colorPrimary'
  | 'colorSecondary'
  | 'sizeSmall'
  | 'thumb'
  | 'track';
export const Switch: React.ComponentType<SwitchProps>;
//export default Switch;
}
declare namespace MaterialUI {
export type TabTypeMap<P = {}, D extends React.ElementType = 'div'> = ExtendButtonBaseTypeMap<{
  props: P & {
    disableFocusRipple?: boolean;
    fullWidth?: boolean;
    icon?: string | React.ReactElement;
    label?: React.ReactNode;
    onChange?: (event: React.ChangeEvent<{ checked: boolean }>, value: any) => void;
    onClick?: React.EventHandler<any>;
    selected?: boolean;
    style?: React.CSSProperties;
    textColor?: string | 'secondary' | 'primary' | 'inherit';
    value?: any;
    wrapped?: boolean;
  };
  defaultComponent: D;
  classKey: TabClassKey;
}>;
export const Tab: ExtendButtonBase<TabTypeMap>;
export type TabClassKey =
  | 'root'
  | 'labelIcon'
  | 'textColorInherit'
  | 'textColorPrimary'
  | 'textColorSecondary'
  | 'selected'
  | 'disabled'
  | 'fullWidth'
  | 'wrapped'
  | 'wrapper';
export type TabProps<
  D extends React.ElementType = TabTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<TabTypeMap<P, D>, D>;
//export default Tab;
}
declare namespace MaterialUI {
export interface TableBodyProps extends StandardProps<TableBodyBaseProps, TableBodyClassKey> {
  component?: React.ElementType<TableBodyBaseProps>;
}
export type TableBodyClassKey = 'root';
export type TableBodyBaseProps = React.HTMLAttributes<HTMLTableSectionElement>;
export const TableBody: React.ComponentType<TableBodyProps>;
//export default TableBody;
}
declare namespace MaterialUI {
export interface TableProps extends StandardProps<TableBaseProps, TableClassKey> {
  component?: React.ElementType<TableBaseProps>;
  padding?: Padding;
  size?: Size;
  stickyHeader?: boolean;
}
export type TableBaseProps = React.TableHTMLAttributes<HTMLTableElement>;
export type Padding = 'default' | 'checkbox' | 'none';
export type Size = 'small' | 'medium';
export type TableClassKey = 'root' | 'stickyHeader';
export const Table: React.ComponentType<TableProps>;
//export default Table;
}
declare namespace MaterialUI {
/**
 * `<TableCell>` will be rendered as an `<th>`or `<td>` depending
 * on the context it is used in. Where context literally is the
 * React `context`.
 *
 * Since it is not decided via prop, we have create loose typings
 * here.
 */
export interface TableCellProps
  extends StandardProps<TableCellBaseProps, TableCellClassKey, 'align'> {
  align?: 'inherit' | 'left' | 'center' | 'right' | 'justify';
  component?: React.ElementType<TableCellBaseProps>;
  padding?: Padding;
  size?: Size;
  sortDirection?: SortDirection;
  variant?: 'head' | 'body' | 'footer';
}
export type TableCellBaseProps = React.ThHTMLAttributes<HTMLTableHeaderCellElement> &
  React.TdHTMLAttributes<HTMLTableDataCellElement>;
export type SortDirection = 'asc' | 'desc' | false;
export type TableCellClassKey =
  | 'root'
  | 'head'
  | 'body'
  | 'footer'
  | 'alignLeft'
  | 'alignCenter'
  | 'alignRight'
  | 'alignJustify'
  | 'sizeSmall'
  | 'paddingCheckbox'
  | 'paddingNone'
  | 'stickyHeader';
export const TableCell: React.ComponentType<TableCellProps>;
//export default TableCell;
}
declare namespace MaterialUI {
export interface TableContainerProps
  extends StandardProps<TableContainerBaseProps, TableContainerClassKey> {
  component?: React.ElementType<TableContainerBaseProps>;
}
export type TableContainerBaseProps = React.HTMLAttributes<HTMLDivElement>;
export type TableContainerClassKey = 'root';
export const TableContainer: React.ComponentType<TableContainerProps>;
//export default TableContainer;
}
declare namespace MaterialUI {
export interface TableFooterProps extends StandardProps<TableFooterBaseProps, TableFooterClassKey> {
  component?: React.ElementType<TableFooterBaseProps>;
}
export type TableFooterClassKey = 'root';
export type TableFooterBaseProps = React.HTMLAttributes<HTMLTableSectionElement>;
export const TableFooter: React.ComponentType<TableFooterProps>;
//export default TableFooter;
}
declare namespace MaterialUI {
export interface TableHeadProps extends StandardProps<TableHeadBaseProps, TableHeadClassKey> {
  component?: React.ElementType<TableHeadBaseProps>;
}
export type TableHeadClassKey = 'root';
export type TableHeadBaseProps = React.HTMLAttributes<HTMLTableSectionElement>;
export const TableHead: React.ComponentType<TableHeadProps>;
//export default TableHead;
}
declare namespace MaterialUI {
export interface TablePaginationActionsProps extends React.HTMLAttributes<HTMLDivElement> {
  backIconButtonProps?: Partial<IconButtonProps>;
  count: number;
  nextIconButtonProps?: Partial<IconButtonProps>;
  onChangePage: (event: React.MouseEvent<HTMLButtonElement> | null, page: number) => void;
  page: number;
  rowsPerPage: number;
}
export const TablePaginationActions: React.ComponentType<TablePaginationActionsProps>;
//export default TablePaginationActions;
}
declare namespace MaterialUI {
export interface LabelDisplayedRowsArgs {
  from: number;
  to: number;
  count: number;
  page: number;
}
export interface TablePaginationTypeMap<P, D extends React.ElementType> {
  props: P &
    TablePaginationBaseProps & {
      ActionsComponent?: React.ElementType<TablePaginationActionsProps>;
      backIconButtonProps?: Partial<IconButtonProps>;
      count: number;
      labelDisplayedRows?: (paginationInfo: LabelDisplayedRowsArgs) => React.ReactNode;
      labelRowsPerPage?: React.ReactNode;
      nextIconButtonProps?: Partial<IconButtonProps>;
      onChangePage: (event: React.MouseEvent<HTMLButtonElement> | null, page: number) => void;
      onChangeRowsPerPage?: React.ChangeEventHandler<HTMLTextAreaElement | HTMLInputElement>;
      page: number;
      rowsPerPage: number;
      rowsPerPageOptions?: Array<number | { value: number; label: string }>;
      SelectProps?: Partial<SelectProps>;
    };
  defaultComponent: D;
  classKey: TablePaginationClassKey;
}
export const TablePagination: OverridableComponent<
  TablePaginationTypeMap<{}, React.ComponentType<TablePaginationBaseProps>>
>;
export type TablePaginationClassKey =
  | 'root'
  | 'toolbar'
  | 'spacer'
  | 'menuItem'
  | 'caption'
  | 'input'
  | 'selectRoot'
  | 'select'
  | 'selectIcon'
  | 'actions';
export type TablePaginationBaseProps = Omit<TableCellProps, 'classes' | 'component'>;
export type TablePaginationProps<
  D extends React.ElementType = React.ComponentType<TablePaginationBaseProps>,
  P = {}
> = OverrideProps<TablePaginationTypeMap<P, D>, D>;
//export default TablePagination;
}
declare namespace MaterialUI {
export interface TableRowProps extends StandardProps<TableRowBaseProps, TableRowClassKey> {
  component?: React.ElementType<TableRowBaseProps>;
  hover?: boolean;
  selected?: boolean;
}
export type TableRowBaseProps = React.HTMLAttributes<HTMLTableRowElement>;
export type TableRowClassKey = 'root' | 'selected' | 'hover' | 'head' | 'footer';
export const TableRow: React.ComponentType<TableRowProps>;
//export default TableRow;
}
declare namespace MaterialUI {
export type TableSortLabelTypeMap<
  P = {},
  D extends React.ElementType = 'span'
> = ExtendButtonBaseTypeMap<{
  props: P & {
    active?: boolean;
    direction?: 'asc' | 'desc';
    hideSortIcon?: boolean;
    IconComponent?: React.ComponentType<{ className: string }>;
  };
  defaultComponent: D;
  classKey: TableSortLabelClassKey;
}>;
export const TableSortLabel: ExtendButtonBase<TableSortLabelTypeMap>;
export type TableSortLabelClassKey =
  | 'root'
  | 'active'
  | 'icon'
  | 'iconDirectionDesc'
  | 'iconDirectionAsc';
export type TableSortLabelProps<
  D extends React.ElementType = TableSortLabelTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<TableSortLabelTypeMap<P, D>, D>;
//export default TableSortLabel;
}
declare namespace MaterialUI {
export interface TabsTypeMap<P = {}, D extends React.ElementType = typeof ButtonBase> {
  props: P & {
    action?: React.Ref<TabsActions>;
    centered?: boolean;
    children?: React.ReactNode;
    indicatorColor?: 'secondary' | 'primary' | string;
    onChange?: (event: React.ChangeEvent<{}>, value: any) => void;
    orientation?: 'horizontal' | 'vertical';
    ScrollButtonComponent?: React.ElementType;
    scrollButtons?: 'auto' | 'desktop' | 'on' | 'off';
    TabIndicatorProps?: Partial<React.HTMLAttributes<HTMLDivElement>>;
    textColor?: 'secondary' | 'primary' | 'inherit' | string;
    value: any;
    variant?: 'standard' | 'scrollable' | 'fullWidth';
    width?: string;
  };
  defaultComponent: D;
  classKey: TabsClassKey;
}
export const Tabs: OverridableComponent<TabsTypeMap>;
export type TabsClassKey =
  | 'root'
  | 'flexContainer'
  | 'scroller'
  | 'fixed'
  | 'scrollable'
  | 'centered'
  | 'scrollButtons'
  | 'scrollButtonsDesktop'
  | 'indicator';
export interface TabsActions {
  updateIndicator(): void;
}
export type TabsProps<
  D extends React.ElementType = TabsTypeMap['defaultComponent'],
  P = {}
> = OverrideProps<TabsTypeMap<P, D>, D>;
//export default Tabs;
}
declare namespace MaterialUI {
export interface BaseTextFieldProps
  extends StandardProps<
    FormControlProps,
    TextFieldClassKey,
    // event handlers are declared on derived interfaces
    'onChange' | 'onBlur' | 'onFocus' | 'defaultValue'
  > {
  autoComplete?: string;
  autoFocus?: boolean;
  color?: 'primary' | 'secondary';
  children?: React.ReactNode;
  defaultValue?: unknown;
  disabled?: boolean;
  error?: boolean;
  FormHelperTextProps?: Partial<FormHelperTextProps>;
  fullWidth?: boolean;
  helperText?: React.ReactNode;
  id?: string;
  InputLabelProps?: Partial<InputLabelProps>;
  inputRef?: React.Ref<any>;
  label?: React.ReactNode;
  margin?: PropTypes.Margin;
  multiline?: boolean;
  name?: string;
  placeholder?: string;
  required?: boolean;
  rows?: string | number;
  rowsMax?: string | number;
  size?: 'small' | 'medium';
  select?: boolean;
  SelectProps?: Partial<SelectProps>;
  type?: string;
  value?: unknown;
}
export interface StandardTextFieldProps extends BaseTextFieldProps {
  onBlur?: StandardInputProps['onBlur'];
  onChange?: StandardInputProps['onChange'];
  onFocus?: StandardInputProps['onFocus'];
  variant?: 'standard';
  InputProps?: Partial<StandardInputProps>;
  inputProps?: StandardInputProps['inputProps'];
}
export interface FilledTextFieldProps extends BaseTextFieldProps {
  onBlur?: FilledInputProps['onBlur'];
  onChange?: FilledInputProps['onChange'];
  onFocus?: FilledInputProps['onFocus'];
  variant: 'filled';
  InputProps?: Partial<FilledInputProps>;
  inputProps?: FilledInputProps['inputProps'];
}
export interface OutlinedTextFieldProps extends BaseTextFieldProps {
  onBlur?: OutlinedInputProps['onBlur'];
  onChange?: OutlinedInputProps['onChange'];
  onFocus?: OutlinedInputProps['onFocus'];
  variant: 'outlined';
  InputProps?: Partial<OutlinedInputProps>;
  inputProps?: OutlinedInputProps['inputProps'];
}
export type TextFieldProps = StandardTextFieldProps | FilledTextFieldProps | OutlinedTextFieldProps;
export type TextFieldClassKey = 'root';
export const TextField: React.ComponentType<TextFieldProps>;
//export default TextField;
}
declare namespace MaterialUI {
export interface ToolbarProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, ToolbarClassKey> {
  component?: React.ElementType<React.HTMLAttributes<HTMLDivElement>>;
  disableGutters?: boolean;
  variant?: 'regular' | 'dense';
}
export type ToolbarClassKey = 'root' | 'gutters' | 'regular' | 'dense';
export const Toolbar: React.ComponentType<ToolbarProps>;
//export default Toolbar;
}
declare namespace MaterialUI {
export type PopperPlacementType =
  | 'bottom-end'
  | 'bottom-start'
  | 'bottom'
  | 'left-end'
  | 'left-start'
  | 'left'
  | 'right-end'
  | 'right-start'
  | 'right'
  | 'top-end'
  | 'top-start'
  | 'top';
export interface PopperProps extends React.HTMLAttributes<HTMLDivElement> {
  anchorEl?: null | ReferenceObject | (() => ReferenceObject);
  children:
    | React.ReactNode
    | ((props: {
        placement: PopperPlacementType;
        TransitionProps?: {
          in: boolean;
          onEnter: () => {};
          onExited: () => {};
        };
      }) => React.ReactNode);
  container?: PortalProps['container'];
  disablePortal?: PortalProps['disablePortal'];
  keepMounted?: boolean;
  modifiers?: object;
  open: boolean;
  placement?: PopperPlacementType;
  popperOptions?: object;
  popperRef?: React.Ref<PopperJs>;
  transition?: boolean;
}
export const Popper: React.ComponentType<PopperProps>;
//export default Popper;
}
declare namespace MaterialUI {
export interface TooltipProps
  extends StandardProps<React.HTMLAttributes<HTMLDivElement>, TooltipClassKey, 'title'> {
  arrow?: boolean;
  children?: React.ReactElement;
  disableFocusListener?: boolean;
  disableHoverListener?: boolean;
  disableTouchListener?: boolean;
  enterDelay?: number;
  enterTouchDelay?: number;
  id?: string;
  interactive?: boolean;
  leaveDelay?: number;
  leaveTouchDelay?: number;
  onClose?: (event: React.ChangeEvent<{}>) => void;
  onOpen?: (event: React.ChangeEvent<{}>) => void;
  open?: boolean;
  placement?:
    | 'bottom-end'
    | 'bottom-start'
    | 'bottom'
    | 'left-end'
    | 'left-start'
    | 'left'
    | 'right-end'
    | 'right-start'
    | 'right'
    | 'top-end'
    | 'top-start'
    | 'top';
  PopperProps?: Partial<PopperProps>;
  title: React.ReactNode;
  TransitionComponent?: React.ComponentType<TransitionProps>;
  TransitionProps?: TransitionProps;
}
export type TooltipClassKey =
  | 'popper'
  | 'popperInteractive'
  | 'popperArrow'
  | 'tooltip'
  | 'tooltipArrow'
  | 'arrow'
  | 'touch'
  | 'tooltipPlacementLeft'
  | 'tooltipPlacementRight'
  | 'tooltipPlacementTop'
  | 'tooltipPlacementBottom';
export const Tooltip: React.ComponentType<TooltipProps>;
//export default Tooltip;
}
declare namespace MaterialUI {
export type Overrides = {
  [Name in keyof ComponentNameToClassKey]?: Partial<StyleRules<ComponentNameToClassKey[Name]>>
};
export interface ComponentNameToClassKey {
  MuiAppBar: AppBarClassKey;
  MuiAvatar: AvatarClassKey;
  MuiBackdrop: BackdropClassKey;
  MuiBadge: BadgeClassKey;
  MuiBottomNavigation: BottomNavigationClassKey;
  MuiBottomNavigationAction: BottomNavigationActionClassKey;
  MuiBreadcrumbs: BreadcrumbsClassKey;
  MuiButton: ButtonClassKey;
  MuiButtonBase: ButtonBaseClassKey;
  MuiButtonGroup: ButtonGroupClassKey;
  MuiCard: CardClassKey;
  MuiCardActionArea: CardActionAreaClassKey;
  MuiCardActions: CardActionsClassKey;
  MuiCardContent: CardContentClassKey;
  MuiCardHeader: CardHeaderClassKey;
  MuiCardMedia: CardMediaClassKey;
  MuiCheckbox: CheckboxClassKey;
  MuiChip: ChipClassKey;
  MuiCircularProgress: CircularProgressClassKey;
  MuiCollapse: CollapseClassKey;
  MuiContainer: ContainerClassKey;
  MuiCssBaseline: CssBaselineClassKey;
  MuiDialog: DialogClassKey;
  MuiDialogActions: DialogActionsClassKey;
  MuiDialogContent: DialogContentClassKey;
  MuiDialogContentText: DialogContentTextClassKey;
  MuiDialogTitle: DialogTitleClassKey;
  MuiDivider: DividerClassKey;
  MuiDrawer: DrawerClassKey;
  MuiExpansionPanel: ExpansionPanelClassKey;
  MuiExpansionPanelActions: ExpansionPanelActionsClassKey;
  MuiExpansionPanelDetails: ExpansionPanelDetailsClassKey;
  MuiExpansionPanelSummary: ExpansionPanelSummaryClassKey;
  MuiFab: FabClassKey;
  MuiFilledInput: FilledInputClassKey;
  MuiFormControl: FormControlClassKey;
  MuiFormControlLabel: FormControlLabelClassKey;
  MuiFormGroup: FormGroupClassKey;
  MuiFormHelperText: FormHelperTextClassKey;
  MuiFormLabel: FormLabelClassKey;
  MuiGrid: GridClassKey;
  MuiGridList: GridListClassKey;
  MuiGridListTile: GridListTileClassKey;
  MuiGridListTileBar: GridListTileBarClassKey;
  MuiIcon: IconClassKey;
  MuiIconButton: IconButtonClassKey;
  MuiInput: InputClassKey;
  MuiInputAdornment: InputAdornmentClassKey;
  MuiInputBase: InputBaseClassKey;
  MuiInputLabel: InputLabelClassKey;
  MuiLinearProgress: LinearProgressClassKey;
  MuiLink: LinkClassKey;
  MuiList: ListClassKey;
  MuiListItem: ListItemClassKey;
  MuiListItemAvatar: ListItemAvatarClassKey;
  MuiListItemIcon: ListItemIconClassKey;
  MuiListItemSecondaryAction: ListItemSecondaryActionClassKey;
  MuiListItemText: ListItemTextClassKey;
  MuiListSubheader: ListSubheaderClassKey;
  MuiMenu: MenuClassKey;
  MuiMenuItem: MenuItemClassKey;
  MuiMobileStepper: MobileStepperClassKey;
  MuiNativeSelect: NativeSelectClassKey;
  MuiOutlinedInput: OutlinedInputClassKey;
  MuiPaper: PaperClassKey;
  MuiPopover: PopoverClassKey;
  MuiRadio: RadioClassKey;
  MuiSelect: SelectClassKey;
  MuiSlider: SliderClassKey;
  MuiSnackbar: SnackbarClassKey;
  MuiSnackbarContent: SnackbarContentClassKey;
  MuiStep: StepClasskey;
  MuiStepButton: StepButtonClasskey;
  MuiStepConnector: StepConnectorClasskey;
  MuiStepContent: StepContentClasskey;
  MuiStepIcon: StepIconClasskey;
  MuiStepLabel: StepLabelClasskey;
  MuiStepper: StepperClasskey;
  MuiSvgIcon: SvgIconClassKey;
  MuiSwitch: SwitchClassKey;
  MuiTab: TabClassKey;
  MuiTable: TableClassKey;
  MuiTableBody: TableBodyClassKey;
  MuiTableCell: TableCellClassKey;
  MuiTableContainer: TableContainerClassKey;
  MuiTableFooter: TableFooterClassKey;
  MuiTableHead: TableHeadClassKey;
  MuiTablePagination: TablePaginationClassKey;
  MuiTableRow: TableRowClassKey;
  MuiTableSortLabel: TableSortLabelClassKey;
  MuiTabs: TabsClassKey;
  MuiTextField: TextFieldClassKey;
  MuiToolbar: ToolbarClassKey;
  MuiTooltip: TooltipClassKey;
  MuiTouchRipple: TouchRippleClassKey;
  MuiTypography: TypographyClassKey;
}
}
declare namespace MaterialUI {
export interface MuiMediaQueryListEvent {
  matches: boolean;
}
export interface MuiMediaQueryList {
  matches: boolean;
  addListener: (listener: MuiMediaQueryListListener) => void;
  removeListener: (listener: MuiMediaQueryListListener) => void;
}
export type MuiMediaQueryListListener = (event: MuiMediaQueryListEvent) => void;
export interface Options {
  defaultMatches?: boolean;
  noSsr?: boolean;
  ssrMatchMedia?: (query: string) => { matches: boolean };
}
export function useMediaQuery<Theme = unknown>(
  query: string | ((theme: Theme) => string),
  options?: Options,
): boolean;
}
declare namespace MaterialUI {
export interface RadioGroupProps
  extends StandardProps<FormGroupProps, RadioGroupClassKey, 'onChange'> {
  name?: string;
  onChange?: (event: React.ChangeEvent<HTMLInputElement>, value: string) => void;
  value?: any;
}
export type RadioGroupClassKey = FormGroupClassKey;
export const RadioGroup: React.ComponentType<RadioGroupProps>;
//export default RadioGroup;
}
declare namespace MaterialUI {
export interface WithWidthOptions {
  withTheme?: boolean;
  noSSR?: boolean;
  initialWidth?: Breakpoint;
  resizeInterval?: number;
}
export interface WithWidth {
  width: Breakpoint;
}
export interface WithWidthProps extends Partial<WithWidth> {
  innerRef?: React.Ref<any>;
}
export function isWidthDown(
  breakpoint: Breakpoint,
  screenWidth: Breakpoint,
  inclusive?: boolean,
): boolean;
export function isWidthUp(
  breakpoint: Breakpoint,
  screenWidth: Breakpoint,
  inclusive?: boolean,
): boolean;
export function withWidth(
  options?: WithWidthOptions,
): PropInjector<WithWidth, WithWidthProps>;
}
declare namespace MaterialUI {
export type ComponentsProps = {
  [Name in keyof ComponentsPropsList]?: Partial<ComponentsPropsList[Name]>
};
export interface ComponentsPropsList {
  MuiAppBar: AppBarProps;
  MuiAvatar: AvatarProps;
  MuiBackdrop: BackdropProps;
  MuiBadge: BadgeProps;
  MuiBottomNavigation: BottomNavigationProps;
  MuiBottomNavigationAction: BottomNavigationActionProps;
  MuiBreadcrumbs: BreadcrumbsProps;
  MuiButton: ButtonProps;
  MuiButtonBase: ButtonBaseProps;
  MuiButtonGroup: ButtonGroupProps;
  MuiCard: CardProps;
  MuiCardActionArea: CardActionAreaProps;
  MuiCardActions: CardActionsProps;
  MuiCardContent: CardContentProps;
  MuiCardHeader: CardHeaderProps;
  MuiCardMedia: CardMediaProps;
  MuiCheckbox: CheckboxProps;
  MuiChip: ChipProps;
  MuiCircularProgress: CircularProgressProps;
  MuiCollapse: CollapseProps;
  MuiContainer: ContainerProps;
  MuiCssBaseline: CssBaselineProps;
  MuiDialog: DialogProps;
  MuiDialogActions: DialogActionsProps;
  MuiDialogContent: DialogContentProps;
  MuiDialogContentText: DialogContentTextProps;
  MuiDialogTitle: DialogTitleProps;
  MuiDivider: DividerProps;
  MuiDrawer: DrawerProps;
  MuiExpansionPanel: ExpansionPanelProps;
  MuiExpansionPanelActions: ExpansionPanelActionsProps;
  MuiExpansionPanelDetails: ExpansionPanelDetailsProps;
  MuiExpansionPanelSummary: ExpansionPanelSummaryProps;
  MuiFab: FabProps;
  MuiFilledInput: FilledInputProps;
  MuiFormControl: FormControlProps;
  MuiFormControlLabel: FormControlLabelProps;
  MuiFormGroup: FormGroupProps;
  MuiFormHelperText: FormHelperTextProps;
  MuiFormLabel: FormLabelProps;
  MuiGrid: GridProps;
  MuiGridList: GridListProps;
  MuiGridListTile: GridListTileProps;
  MuiGridListTileBar: GridListTileBarProps;
  MuiIcon: IconProps;
  MuiIconButton: IconButtonProps;
  MuiInput: InputProps;
  MuiInputAdornment: InputAdornmentProps;
  MuiInputBase: InputBaseProps;
  MuiInputLabel: InputLabelProps;
  MuiLinearProgress: LinearProgressProps;
  MuiLink: LinkProps;
  MuiList: ListProps;
  MuiListItem: ListItemProps;
  MuiListItemAvatar: ListItemAvatarProps;
  MuiListItemIcon: ListItemIconProps;
  MuiListItemSecondaryAction: ListItemSecondaryActionProps;
  MuiListItemText: ListItemTextProps;
  MuiListSubheader: ListSubheaderProps;
  MuiMenu: MenuProps;
  MuiMenuItem: MenuItemProps;
  MuiMenuList: MenuListProps;
  MuiMobileStepper: MobileStepperProps;
  MuiModal: ModalProps;
  MuiNativeSelect: NativeSelectProps;
  MuiOutlinedInput: OutlinedInputProps;
  MuiPaper: PaperProps;
  MuiPopover: PopoverProps;
  MuiRadio: RadioProps;
  MuiRadioGroup: RadioGroupProps;
  MuiSelect: SelectProps;
  MuiSlider: SliderProps;
  MuiSnackbar: SnackbarProps;
  MuiSnackbarContent: SnackbarContentProps;
  MuiStep: StepProps;
  MuiStepButton: StepButtonProps;
  MuiStepConnector: StepConnectorProps;
  MuiStepContent: StepContentProps;
  MuiStepIcon: StepIconProps;
  MuiStepLabel: StepLabelProps;
  MuiStepper: StepperProps;
  MuiSvgIcon: SvgIconProps;
  MuiSwitch: SwitchProps;
  MuiTab: TabProps;
  MuiTable: TableProps;
  MuiTableBody: TableBodyProps;
  MuiTableCell: TableCellProps;
  MuiTableContainer: TableContainerProps;
  MuiTableHead: TableHeadProps;
  MuiTablePagination: TablePaginationProps;
  MuiTableRow: TableRowProps;
  MuiTableSortLabel: TableSortLabelProps;
  MuiTabs: TabsProps;
  MuiTextField: TextFieldProps;
  MuiToolbar: ToolbarProps;
  MuiTooltip: TooltipProps;
  MuiTouchRipple: TouchRippleProps;
  MuiTypography: TypographyProps;
  MuiUseMediaQuery: useMediaQueryOptions;
  MuiWithWidth: WithWidthOptions;
}
}
declare namespace MaterialUI {
export type Direction = 'ltr' | 'rtl';
export interface ThemeOptions {
  shape?: ShapeOptions;
  breakpoints?: BreakpointsOptions;
  direction?: Direction;
  mixins?: MixinsOptions;
  overrides?: Overrides;
  palette?: PaletteOptions;
  props?: ComponentsProps;
  shadows?: Shadows;
  spacing?: SpacingOptions;
  transitions?: TransitionsOptions;
  typography?: TypographyOptions | ((palette: Palette) => TypographyOptions);
  zIndex?: ZIndexOptions;
}
export interface Theme {
  shape: Shape;
  breakpoints: Breakpoints;
  direction: Direction;
  mixins: Mixins;
  overrides?: Overrides;
  palette: Palette;
  props?: ComponentsProps;
  shadows: Shadows;
  spacing: Spacing;
  transitions: Transitions;
  typography: Typography;
  zIndex: ZIndex;
}
export function createMuiTheme(options?: ThemeOptions, ...args: object[]): Theme;
}
declare namespace MaterialUI {
export interface FadeProps extends TransitionProps {
  ref?: React.Ref<unknown>;
  theme?: Theme;
}
export const Fade: React.ComponentType<FadeProps>;
//export default Fade;
}
declare namespace MaterialUI {
export interface BackdropProps
  extends StandardProps<
    React.HTMLAttributes<HTMLDivElement> & Partial<FadeProps>,
    BackdropClassKey
  > {
  /**
   * If `true`, the backdrop is invisible.
   * It can be used when rendering a popover or a custom select component.
   */
  invisible?: boolean;
  /**
   * If `true`, the backdrop is open.
   */
  open: boolean;
  /**
   * The duration for the transition, in milliseconds.
   * You may specify a single timeout for all transitions, or individually with an object.
   */
  transitionDuration?: TransitionProps['timeout'];
}
export type BackdropClassKey = 'root' | 'invisible';
export function Backdrop(props: BackdropProps): JSX.Element;
}
declare namespace MaterialUI {
type BoxStyleFunction = ComposedStyleFunction<
  [
    typeof borders,
    typeof display,
    typeof flexbox,
    typeof palette,
    typeof positions,
    typeof shadows,
    typeof sizing,
    typeof spacing,
    typeof typography
  ]
>;
type SystemProps = PropsFor<BoxStyleFunction>;
type ElementProps = Omit<React.HTMLAttributes<HTMLElement>, keyof SystemProps>;
export interface BoxProps extends ElementProps, SystemProps {
  // styled API
  component?: React.ElementType;
  clone?: boolean;
  // workaround for https://github.com/mui-org/material-ui/pull/15611
  css?: SystemProps;
}
export const Box: React.ComponentType<BoxProps>;
//export default Box;
}
declare namespace MaterialUI {
export interface GrowProps extends Omit<TransitionProps, 'timeout'> {
  ref?: React.Ref<unknown>;
  theme?: Theme;
  timeout?: TransitionProps['timeout'] | 'auto';
}
export const Grow: React.ComponentType<GrowProps>;
//export default Grow;
}
declare namespace MaterialUI {
export interface HiddenProps {
  implementation?: 'js' | 'css';
  initialWidth?: Breakpoint;
  lgDown?: boolean;
  lgUp?: boolean;
  mdDown?: boolean;
  mdUp?: boolean;
  only?: Breakpoint | Breakpoint[];
  smDown?: boolean;
  smUp?: boolean;
  xlDown?: boolean;
  xlUp?: boolean;
  xsDown?: boolean;
  xsUp?: boolean;
}
export const Hidden: React.ComponentType<HiddenProps>;
//export default Hidden;
}
declare namespace MaterialUI {
export interface NoSsrProps {
  children?: React.ReactNode;
  defer?: boolean;
  fallback?: React.ReactNode;
}
export const NoSsr: React.ComponentType<NoSsrProps>;
//export default NoSsr;
}
declare namespace MaterialUI {
export interface RootRefProps<T = any> {
  rootRef?: ((instance: T | null) => void) | React.RefObject<T>;
}
export const RootRef: React.ComponentType<RootRefProps>;
//export default RootRef;
}
declare namespace MaterialUI {
export interface SwipeableDrawerProps extends Omit<DrawerProps, 'onClose' | 'open'> {
  disableBackdropTransition?: boolean;
  disableDiscovery?: boolean;
  disableSwipeToOpen?: boolean;
  hysteresis?: number;
  minFlingVelocity?: number;
  onClose: React.ReactEventHandler<{}>;
  onOpen: React.ReactEventHandler<{}>;
  open: boolean;
  SwipeAreaProps?: object;
  swipeAreaWidth?: number;
}
export const SwipeableDrawer: React.ComponentType<SwipeableDrawerProps>;
//export default SwipeableDrawer;
}
declare namespace MaterialUI {
export interface TextareaAutosizeProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  rowsMax?: string | number;
  rowsMin?: string | number;
}
export const TextareaAutosize: React.ComponentType<
  TextareaAutosizeProps & {
    ref?: React.Ref<HTMLTextAreaElement>;
  }
>;
//export default TextareaAutosize;
}
declare namespace MaterialUI {
export interface UseScrollTriggerOptions {
  disableHysteresis?: boolean;
  target?: Node | Window;
  threshold?: number;
}
export function useScrollTrigger(options?: UseScrollTriggerOptions): boolean;
}
declare namespace MaterialUI {
export interface WithWidthOptions {
  withTheme?: boolean;
  noSSR?: boolean;
  initialWidth?: Breakpoint;
  resizeInterval?: number;
}
export interface WithWidth {
  width: Breakpoint;
}
export interface WithWidthProps extends Partial<WithWidth> {
  innerRef?: React.Ref<any>;
}
export function isWidthDown(
  breakpoint: Breakpoint,
  screenWidth: Breakpoint,
  inclusive?: boolean,
): boolean;
export function isWidthUp(
  breakpoint: Breakpoint,
  screenWidth: Breakpoint,
  inclusive?: boolean,
): boolean;
export function withWidth(
  options?: WithWidthOptions,
): PropInjector<WithWidth, WithWidthProps>;
}
declare namespace MaterialUI {
export interface WithMobileDialogOptions {
  breakpoint: Breakpoint;
}
export interface WithMobileDialog extends WithWidth {
  fullScreen: boolean;
}
/**
 * @deprecated
 */
export interface InjectedProps extends WithMobileDialog {}
export function withMobileDialog<P = {}>(
  options?: WithMobileDialogOptions,
): PropInjector<WithMobileDialog, Partial<WithMobileDialog>>;
}
declare namespace MaterialUI {
export interface ZoomProps extends TransitionProps {
  ref?: React.Ref<unknown>;
  theme?: Theme;
}
export const Zoom: React.ComponentType<ZoomProps>;
//export default Zoom;
}
declare namespace MaterialUI {
//export default createStyles;
}
declare namespace MaterialUI {
/**
 * `makeStyles` where the passed `styles` do not depend on props
 */
export function makeStyles<Theme = DefaultTheme, ClassKey extends string = string>(
  style: Styles<Theme, {}, ClassKey>,
  options?: Omit<WithStylesOptions<Theme>, 'withTheme'>,
): (props?: any) => ClassNameMap<ClassKey>;
/**
 * `makeStyles` where the passed `styles` do depend on props
 */
export function makeStyles<
  Theme = DefaultTheme,
  Props extends {} = {},
  ClassKey extends string = string
>(
  styles: Styles<Theme, Props, ClassKey>,
  options?: Omit<WithStylesOptions<Theme>, 'withTheme'>,
): (props: Props) => ClassNameMap<ClassKey>;
}
declare namespace MaterialUI {
export interface ResponsiveFontSizesOptions {
  breakpoints?: Breakpoint[];
  disableAlign?: boolean;
  factor?: number;
  variants?: Variant[];
}
export function responsiveFontSizes(
  theme: Theme,
  options?: ResponsiveFontSizesOptions,
): Theme;
}
declare namespace MaterialUI {
export function useTheme<T = Theme>(): T;
}
declare namespace MaterialUI {
export interface WithTheme {
  theme: Theme;
}
export interface ThemedComponentProps extends Partial<WithTheme> {
  /**
   * Deprecated. Will be removed in v5. Refs are now automatically forwarded to
   * the inner component.
   * @deprecated since version 4.0
   */
  innerRef?: React.Ref<any>;
  ref?: React.Ref<unknown>;
}
export const withTheme: PropInjector<WithTheme, ThemedComponentProps>;
//export default withTheme;
}
declare namespace MaterialUI {
export interface WithTheme {
  theme: Theme;
}
export interface ThemedComponentProps extends Partial<WithTheme> {
  /**
   * Deprecated. Will be removed in v5. Refs are now automatically forwarded to
   * the inner component.
   * @deprecated since version 4.0
   */
  innerRef?: React.Ref<any>;
  ref?: React.Ref<unknown>;
}
export const withTheme: PropInjector<WithTheme, ThemedComponentProps>;
//export default withTheme;
}
declare namespace MaterialUI {
// These definitions are almost identical to the ones in @material-ui/styles/styled
// Only difference is that ComponentCreator has a default theme type
// If you need to change these types, update the ones in @material-ui/styles as well
/**
 * @internal
 */
export type ComponentCreator<Component extends React.ElementType> = <
  Theme = DefaultTheme,
  Props extends {} = {}
>(
  styles:
    | CreateCSSProperties<Props>
    | ((props: { theme: Theme } & Props) => CreateCSSProperties<Props>),
  options?: WithStylesOptions<Theme>,
) => React.ComponentType<
  Omit<
    JSX.LibraryManagedAttributes<Component, React.ComponentProps<Component>>,
    'classes' | 'className'
  > &
    StyledComponentProps<'root'> & { className?: string } & (Props extends { theme: Theme }
      ? Omit<Props, 'theme'> & { theme?: Theme }
      : Props)
>;
export interface StyledProps {
  className: string;
}
export function styled<Component extends React.ElementType>(
  Component: Component,
): ComponentCreator<Component>;
}
declare namespace MaterialUI {
export interface GenerateClassNameOptions {
  disableGlobal?: boolean;
  productionPrefix?: string;
  seed?: string;
}
export function createGenerateClassName(options?: GenerateClassNameOptions): GenerateId;
}
declare namespace MaterialUI {
/**
 * This function doesn't really "do anything" at runtime, it's just the identity
 * function. Its only purpose is to defeat TypeScript's type widening when providing
 * style rules to `withStyles` which are a function of the `Theme`.
 *
 * @param styles a set of style mappings
 * @returns the same styles that were passed in
 */
// For TypeScript v3.5 Props has to extend {} instead of object
// See https://github.com/mui-org/material-ui/issues/15942
// and https://github.com/microsoft/TypeScript/issues/31735
export function createStyles<ClassKey extends string, Props extends {}>(
  styles: StyleRules<Props, ClassKey>,
): StyleRules<Props, ClassKey>;
}
declare namespace MaterialUI {
interface ThemeWithProps<Components> {
  props?: { [K in keyof Components]: Partial<Components[K]> };
}
type ThemedProps<Theme, Name extends keyof any> = Theme extends { props: Record<Name, infer Props> }
  ? Props
  : {};
export function getThemeProps<
  Theme extends ThemeWithProps<any>,
  Props,
  Name extends keyof any
>(params: { props: Props; name: Name; theme?: Theme }): Props & ThemedProps<Theme, Name>;
}
declare namespace MaterialUI {
export function jssPreset(): JssOptions;
}
declare namespace MaterialUI {
/**
 * `makeStyles` where the passed `styles` do not depend on props
 */
export function makeStyles<Theme = DefaultTheme, ClassKey extends string = string>(
  style: Styles<Theme, {}, ClassKey>,
  options?: Omit<WithStylesOptions<Theme>, 'withTheme'>,
): (props?: any) => ClassNameMap<ClassKey>;
/**
 * `makeStyles` where the passed `styles` do depend on props
 */
export function makeStyles<
  Theme = DefaultTheme,
  Props extends {} = {},
  ClassKey extends string = string
>(
  styles: Styles<Theme, Props, ClassKey>,
  options?: Omit<WithStylesOptions<Theme>, 'withTheme'>,
): (props: Props) => ClassNameMap<ClassKey>;
}
declare namespace MaterialUI {
export interface Classes {
  [k: string]: string;
}
export interface MergeClassesOption {
  baseClasses: Classes;
  newClasses: Classes;
  Component: React.ElementType;
}
export function mergeClasses(options?: MergeClassesOption): Classes;
}
declare namespace MaterialUI {
export interface StylesOptions {
  disableGeneration?: boolean;
  generateClassName?: GenerateId;
  injectFirst?: boolean;
  jss?: Jss;
  // TODO need info @oliviertassinari
  sheetsCache?: {};
  // TODO need info @oliviertassinari
  sheetsManager?: {};
  // TODO need info @oliviertassinari
  sheetsRegistry?: {};
}
export const StylesContext: React.Context<StylesOptions>;
export interface StylesProviderProps extends StylesOptions {
  children?: React.ReactNode;
}
export const StylesProvider: React.ComponentType<StylesProviderProps>;
//export default StylesProvider;
}
declare namespace MaterialUI {
declare class ServerStyleSheets {
  constructor(options?: object);
  collect(children?: React.ReactNode, options?: object): React.ReactElement<StylesProviderProps>;
  toString(): string;
  getStyleElement(props?: object): React.ReactElement;
}
//export default ServerStyleSheets;
}
declare namespace MaterialUI {
/**
 * @internal
 */
export type ComponentCreator<Component extends React.ElementType> = <
  Theme = DefaultTheme,
  Props extends {} = {}
>(
  styles:
    | CreateCSSProperties<Props>
    | ((props: { theme: Theme } & Props) => CreateCSSProperties<Props>),
  options?: WithStylesOptions<Theme>,
) => React.ComponentType<
  Omit<
    JSX.LibraryManagedAttributes<Component, React.ComponentProps<Component>>,
    'classes' | 'className'
  > &
    StyledComponentProps<'root'> & { className?: string } & (Props extends { theme: Theme }
      ? Omit<Props, 'theme'> & { theme?: Theme }
      : Props)
>;
export interface StyledProps {
  className: string;
}
export function styled<Component extends React.ElementType>(
  Component: Component,
): ComponentCreator<Component>;
}
declare namespace MaterialUI {
export interface ThemeProviderProps<Theme = DefaultTheme> {
  children?: React.ReactNode;
  theme: Partial<Theme> | ((outerTheme: Theme) => Theme);
}
export function MuiThemeProvider<T = DefaultTheme>(
  props: ThemeProviderProps<T>,
): React.ReactElement<ThemeProviderProps<T>>;
}
declare namespace MaterialUI {
export function useTheme<T = DefaultTheme>(): T;
}
declare namespace MaterialUI {
// Disable automatic export
type JSSFontface = CSS.FontFace & { fallbacks?: CSS.FontFace[] };
/**
 * Allows the user to augment the properties available
 */
export interface BaseCSSProperties extends CSS.Properties<number | string> {
  '@font-face'?: JSSFontface | JSSFontface[];
}
export interface CSSProperties extends BaseCSSProperties {
  // Allow pseudo selectors and media queries
  // `unknown` is used since TS does not allow assigning an interface without
  // an index signature to one with an index signature. This is to allow type safe
  // module augmentation.
  // Technically we want any key not typed in `BaseCSSProperties` to be of type
  // `CSSProperties` but this doesn't work. The index signature needs to cover
  // BaseCSSProperties as well. Usually you would use `BaseCSSProperties[keyof BaseCSSProperties]`
  // but this would not allow assigning React.CSSProperties to CSSProperties
  [k: string]: unknown | CSSProperties;
}
export type BaseCreateCSSProperties<Props extends object = {}> = {
  [P in keyof BaseCSSProperties]: BaseCSSProperties[P] | ((props: Props) => BaseCSSProperties[P])
};
export interface CreateCSSProperties<Props extends object = {}>
  extends BaseCreateCSSProperties<Props> {
  // Allow pseudo selectors and media queries
  [k: string]:
    | BaseCreateCSSProperties<Props>[keyof BaseCreateCSSProperties<Props>]
    | CreateCSSProperties<Props>;
}
/**
 * This is basically the API of JSS. It defines a Map<string, CSS>,
 * where
 * - the `keys` are the class (names) that will be created
 * - the `values` are objects that represent CSS rules (`React.CSSProperties`).
 *
 * if only `CSSProperties` are matched `Props` are inferred to `any`
 */
export type StyleRules<Props extends object = {}, ClassKey extends string = string> = Record<
  ClassKey,
  // JSS property bag
  | CSSProperties
  // JSS property bag where values are based on props
  | CreateCSSProperties<Props>
  // JSS property bag based on props
  | ((props: Props) => CreateCSSProperties<Props>)
>;
/**
 * @internal
 */
export type StyleRulesCallback<Theme, Props extends object, ClassKey extends string = string> = (
  theme: Theme,
) => StyleRules<Props, ClassKey>;
export type Styles<Theme, Props extends object, ClassKey extends string = string> =
  | StyleRules<Props, ClassKey>
  | StyleRulesCallback<Theme, Props, ClassKey>;
export interface WithStylesOptions<Theme = DefaultTheme> extends JSS.StyleSheetFactoryOptions {
  defaultTheme?: Theme;
  flip?: boolean;
  withTheme?: boolean;
  name?: string;
}
export type ClassNameMap<ClassKey extends string = string> = Record<ClassKey, string>;
/**
 * @internal
 */
export type ClassKeyInferable<Theme, Props extends object> = string | Styles<Theme, Props>;
export type ClassKeyOfStyles<StylesOrClassKey> = StylesOrClassKey extends string
  ? StylesOrClassKey
  : StylesOrClassKey extends StyleRulesCallback<any, any, infer ClassKey>
  ? ClassKey
  : StylesOrClassKey extends StyleRules<any, infer ClassKey>
  ? ClassKey
  : never;
/**
 * infers the type of the props used in the styles
 */
export type PropsOfStyles<StylesType> = StylesType extends Styles<any, infer Props> ? Props : {};
/**
 * infers the type of the theme used in the styles
 */
export type ThemeOfStyles<StylesType> = StylesType extends Styles<infer Theme, any> ? Theme : {};
export type WithStyles<
  StylesType extends ClassKeyInferable<any, any>,
  IncludeTheme extends boolean | undefined = false
> = (IncludeTheme extends true ? { theme: ThemeOfStyles<StylesType> } : {}) & {
  classes: ClassNameMap<ClassKeyOfStyles<StylesType>>;
  innerRef?: React.Ref<any>;
} & PropsOfStyles<StylesType>;
export interface StyledComponentProps<ClassKey extends string = string> {
  /**
   * Override or extend the styles applied to the component.
   */
  classes?: Partial<ClassNameMap<ClassKey>>;
  innerRef?: React.Ref<any>;
}
export function withStyles<
  StylesType extends Styles<any, any>,
  Options extends WithStylesOptions<ThemeOfStyles<StylesType>> = {}
>(
  style: StylesType,
  options?: Options,
): PropInjector<
  WithStyles<StylesType, Options['withTheme']>,
  StyledComponentProps<ClassKeyOfStyles<StylesType>> & PropsOfStyles<StylesType>
>;
}
declare namespace MaterialUI {
export interface WithThemeCreatorOption<Theme = DefaultTheme> {
  defaultTheme?: Theme;
}
export interface WithTheme<Theme = DefaultTheme> {
  theme: Theme;
  /**
   * Deprecated. Will be removed in v5. Refs are now automatically forwarded to
   * the inner component.
   * @deprecated since version 4.0
   */
  innerRef?: React.Ref<any>;
}
export interface ThemedComponentProps extends Partial<WithTheme> {
  ref?: React.Ref<unknown>;
}
export function withThemeCreator<Theme = DefaultTheme>(
  option?: WithThemeCreatorOption<Theme>,
): PropInjector<WithTheme<Theme>, ThemedComponentProps>;
export function withTheme<
  Theme,
  C extends React.ComponentType<ConsistentWith<React.ComponentProps<C>, WithTheme<Theme>>>
>(
  component: C,
): React.ComponentType<
  Omit<JSX.LibraryManagedAttributes<C, React.ComponentProps<C>>, keyof WithTheme<Theme>> &
    Partial<WithTheme<Theme>> &
    ThemedComponentProps
>;
}

declare namespace MaterialUI {
  export const colors: {
    amber: Color;
    blue: Color;
    blueGrey: Color;
    brown: Color;
    common: Color;
    cyan: Color;
    deepOrange: Color;
    deepPurple: Color;
    green: Color;
    grey: Color;
    indigo: Color;
    lightBlue: Color;
    lightGreen: Color;
    lime: Color;
    orange: Color;
    pink: Color;
    purple: Color;
    red: Color;
    teal: Color;
    yellow: Color;
  }
}