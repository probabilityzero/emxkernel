#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]
#![feature(alloc_error_handler)]
#![feature(const_mut_refs)]

extern crate alloc;

use core::panic::PanicInfo;
use lazy_static::lazy_static;
use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame};
use x86_64::structures::gdt::{GlobalDescriptorTable, Descriptor, SegmentSelector};
use x86_64::registers::segmentation::{Segment, CS, DS, ES, FS, GS, SS};
use x86_64::instructions::port::Port;
use x86_64::VirtAddr;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::boxed::Box;
use core::fmt;
use core::fmt::Write;
use bootloader::BootInfo;
use x86_64::structures::paging::{
    OffsetPageTable, PageTable,
    FrameAllocator, PhysFrame, Size4KiB,
    Page, Mapper, PageTableFlags,
};
use core::sync::atomic::{AtomicU64, Ordering};
use x86_64::registers::control::{Cr3, Cr0, Cr0Flags};
use x86_64::PhysAddr;
use core::arch::asm;

// --- Panic Handler ---

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    println!("{}", info);
    loop {}
}

// --- VGA Text Mode Output ---

const VGA_WIDTH: usize = 80;
const VGA_HEIGHT: usize = 25;

#[repr(transparent)]
struct VgaBuffer([[u16; VGA_WIDTH]; VGA_HEIGHT]);

static mut VGA_BUFFER: *mut VgaBuffer = 0xb8000 as *mut VgaBuffer;
static mut VGA_CURSOR_X: usize = 0;
static mut VGA_CURSOR_Y: usize = 0;

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => ($crate::_print(format_args!($($arg)*)));
}

#[macro_export]
macro_rules! println {
    () => ($crate::print!("\n"));
    ($($arg:tt)*) => ($crate::print!("{}\n", format_args!($($arg)*)));
}

#[doc(hidden)]
pub fn _print(args: fmt::Arguments) {
    unsafe {
        VGA_WRITER.write_fmt(args).unwrap();
    }
}

// --- VGA Writer ---

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    Black = 0,
    Blue = 1,
    Green = 2,
    Cyan = 3,
    Red = 4,
    Magenta = 5,
    Brown = 6,
    LightGray = 7,
    DarkGray = 8,
    LightBlue = 9,
    LightGreen = 10,
    LightCyan = 11,
    LightRed = 12,
    Pink = 13,
    Yellow = 14,
    White = 15,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
struct ColorCode(u8);

impl ColorCode {
    fn new(foreground: Color, background: Color) -> ColorCode {
        ColorCode((background as u8) << 4 | (foreground as u8))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
struct ScreenChar {
    ascii_character: u8,
    color_code: ColorCode,
}

struct Writer {
    column_position: usize,
    color_code: ColorCode,
    buffer: &'static mut VgaBuffer,
}

impl Writer {
    pub fn write_byte(&mut self, byte: u8) {
        match byte {
            b'\n' => self.new_line(),
            byte => {
                if self.column_position >= VGA_WIDTH {
                    self.new_line();
                }

                let row = VGA_HEIGHT - 1;
                let col = self.column_position;

                let color_code = self.color_code;
                self.buffer.0[row][col] = ((color_code.0 as u16) << 8) | (byte as u16);
                self.column_position += 1;
            }
        }
    }

    fn new_line(&mut self) {
        for row in 1..VGA_HEIGHT {
            for col in 0..VGA_WIDTH {
                unsafe {
                    let character = self.buffer.0[row][col];
                    self.buffer.0[row - 1][col] = character;
                }
            }
        }
        self.clear_row(VGA_HEIGHT - 1);
        self.column_position = 0;
    }


    fn clear_row(&mut self, row: usize) {
        unsafe {
            let blank = ((self.color_code.0 as u16) << 8) | (b' ' as u16);
            for col in 0..VGA_WIDTH {
                self.buffer.0[row][col] = blank;
            }
        }
    }

    pub fn write_string(&mut self, s: &str) {
        for byte in s.bytes() {
            match byte {
                // printable ASCII byte or newline
                0x20..=0x7e | b'\n' => self.write_byte(byte),
                // not part of printable ASCII range
                _ => self.write_byte(0xfe),
            }
        }
    }
}

impl fmt::Write for Writer {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.write_string(s);
        Ok(())
    }
}

lazy_static! {
    pub static ref VGA_WRITER: Writer = Writer {
        column_position: 0,
        color_code: ColorCode::new(Color::Yellow, Color::Black),
        buffer: unsafe { &mut *(0xb8000 as *mut VgaBuffer) },
    };
}

fn print_char(c: u8) {
    unsafe {
        let color_code = ((Color::White as u16) << 4) | (Color::Black as u16); // White on Black
        let character = (color_code << 8) | (c as u16);

        if c == b'\n' {
            VGA_CURSOR_X = 0;
            VGA_CURSOR_Y += 1;
        } else {
            (*VGA_BUFFER).0[VGA_CURSOR_Y][VGA_CURSOR_X] = character;
            VGA_CURSOR_X += 1;
        }

        if VGA_CURSOR_X >= VGA_WIDTH {
            VGA_CURSOR_X = 0;
            VGA_CURSOR_Y += 1;
        }

        if VGA_CURSOR_Y >= VGA_HEIGHT {
            for y in 1..VGA_HEIGHT {
                for x in 0..VGA_WIDTH {
                    (*VGA_BUFFER).0[y - 1][x] = (*VGA_BUFFER).0[y][x];
                }
            }

            for x in 0..VGA_WIDTH {
                (*VGA_BUFFER).0[VGA_HEIGHT - 1][x] = (color_code << 8) | (b' ' as u16);
            }
            VGA_CURSOR_Y = VGA_HEIGHT - 1;
        }
        update_cursor(VGA_CURSOR_X as u16, VGA_CURSOR_Y as u16);
    }
}

// --- GDT ---
lazy_static! {
    static ref GDT: (GlobalDescriptorTable, Selectors) = {
        let mut gdt = GlobalDescriptorTable::new();
        let code_selector = gdt.add_entry(Descriptor::kernel_code_segment());
        let tss_selector = gdt.add_entry(Descriptor::tss_segment(&TSS));
        let null_selector = gdt.add_entry(Descriptor::UserSegment(0)); // Null segment
        (gdt, Selectors { code_selector, tss_selector, null_selector })
    };
}

struct Selectors {
    code_selector: SegmentSelector,
    tss_selector: SegmentSelector,
    null_selector: SegmentSelector, // Null segment
}

pub fn init_gdt() {
    use x86_64::instructions::tables::load_tss;

    GDT.0.load();
    unsafe {
        CS::set_reg(GDT.1.code_selector);
        load_tss(GDT.1.tss_selector);
        // Set other segment registers to the null segment
        DS::set_reg(GDT.1.null_selector);
        ES::set_reg(GDT.1.null_selector);
        FS::set_reg(GDT.1.null_selector);
        GS::set_reg(GDT.1.null_selector);
        SS::set_reg(GDT.1.null_selector);
    }
}

// --- TSS ---
use x86_64::structures::tss::TaskStateSegment;

lazy_static! {
    static ref TSS: TaskStateSegment = {
        let mut tss = TaskStateSegment::new();
        tss.interrupt_stack_table[DOUBLE_FAULT_IST_INDEX as usize] = {
            const STACK_SIZE: usize = 4096 * 5;
            static mut STACK: [u8; STACK_SIZE] = [0; STACK_SIZE];

            let stack_start = VirtAddr::from_ptr(unsafe { &STACK });
            let stack_end = stack_start + STACK_SIZE;
            stack_end
        };
        tss
    };
}
pub const DOUBLE_FAULT_IST_INDEX: u16 = 0;

// --- Interrupts ---

lazy_static! {
    static ref IDT: InterruptDescriptorTable = {
        let mut idt = InterruptDescriptorTable::new();
        idt.breakpoint.set_handler_fn(breakpoint_handler);
        unsafe {
            idt.double_fault
                .set_handler_fn(double_fault_handler)
                .set_stack_index(DOUBLE_FAULT_IST_INDEX);
        }
        idt[0x20].set_handler_fn(timer_interrupt_handler); // PIC timer
        idt[0x21].set_handler_fn(keyboard_interrupt_handler); // Keyboard
        idt[0x80].set_handler_fn(syscall_handler); // System calls
        idt
    };
}

pub fn init_idt() {
    IDT.load();
}

extern "x86-interrupt" fn breakpoint_handler(stack_frame: InterruptStackFrame) {
    println!("EXCEPTION: BREAKPOINT\n{:#?}", stack_frame);
}

extern "x86-interrupt" fn double_fault_handler(
    stack_frame: InterruptStackFrame,
    _error_code: u64,
) -> ! {
    println!("EXCEPTION: DOUBLE FAULT\n{:#?}", stack_frame);
    println!("Halting...");
    loop {
        x86_64::instructions::hlt();
    }
}

// --- System Calls ---

const SYSCALL_PRINT: u64 = 1;
const SYSCALL_EXIT: u64 = 2;
const SYSCALL_YIELD: u64 = 3;
const SYSCALL_ALLOCATE: u64 = 4;
const SYSCALL_DEALLOCATE: u64 = 5;

extern "x86-interrupt" fn syscall_handler(stack_frame: &mut InterruptStackFrame) {
    // Correctly get syscall number and arguments from registers
    let (syscall_number, arg1, arg2, arg3) = unsafe {
        (
            stack_frame.general.rax, // System call number in RAX
            stack_frame.general.rdi, // Arg 1 in RDI
            stack_frame.general.rsi, // Arg 2 in RSI
            stack_frame.general.rdx, // Arg 3 in RDX
        )
    };


    match syscall_number {
        SYSCALL_PRINT => {
            // Placeholder for printing (still needs proper memory handling)
            println!("syscall called");
        }
        SYSCALL_EXIT => {
            println!("syscall exit");
            // switch_to_next_task(stack_frame); // Corrected call

        }
        SYSCALL_YIELD => {
            println!("syscall yeild");
            //switch_to_next_task(stack_frame);  // Corrected call
        }
        SYSCALL_ALLOCATE => {
            // Placeholder for allocation
            println!("syscall allocate");
        }
        SYSCALL_DEALLOCATE => {
            // Placeholder for deallocation
            println!("syscall deallocate");
        }
        _ => {
            println!("Unknown system call: {}", syscall_number);
        }
    }
    // Send End of Interrupt (EOI) to the PIC
    unsafe {
        let mut pic1_command_port: Port<u8> = Port::new(0x20);
        pic1_command_port.write(0x20); // EOI command
    }
}


// --- Command Line Interface ---

struct CommandLine {
    buffer: Vec<u8>,
}

impl CommandLine {
    fn new() -> CommandLine {
        CommandLine { buffer: Vec::new() }
    }

    fn add_char(&mut self, c: char) {
        if c == '\x08' { // Backspace
            if !self.buffer.is_empty() {
                self.buffer.pop();
                print!("\x08 \x08"); // Backspace, space, backspace (to clear the character)
            }
        } else if c == '\n' {
            self.execute_command();
            self.buffer.clear();
            print!("\n> ");
        } else {
            self.buffer.push(c as u8);
            print!("{}", c);
        }
    }

    fn execute_command(&mut self) {
        let command_str = String::from_utf8_lossy(&self.buffer);
        let command_str = command_str.trim();

        match command_str {
            "help" => {
                println!("\nAvailable commands:");
                println!("  help: Show this help message");
                println!("  clear: Clear the screen");
                println!("  echo <text>: Print the given text");
                println!("  exit: Exit the kernel (switches to the next task)");
                println!(" tasks : Show the tasks");
            }
            "clear" => {
                unsafe {
                    for y in 0..VGA_HEIGHT {
                        for x in 0..VGA_WIDTH {
                            (*VGA_BUFFER).0[y][x] = 0;
                        }
                    }
                }
                VGA_CURSOR_X = 0;
                VGA_CURSOR_Y = 0;
                update_cursor(VGA_CURSOR_X as u16, VGA_CURSOR_Y as u16);
            }
            "exit" => {
                println!("exit");
                // switch_to_next_task();
            }
            "tasks" => {
                println!("tasks");
                // print_task_list();
            }

            _ => {
                if command_str.starts_with("echo ") {
                    let text = &command_str[5..]; // Extract the text after "echo "
                    println!("\n{}", text);
                } else if !command_str.is_empty() {
                    println!("\nUnknown command: {}", command_str);
                }
            }
        }
    }
}

lazy_static! {
    static ref COMMAND_LINE: spin::Mutex<CommandLine> = spin::Mutex::new(CommandLine::new());
}

extern "x86-interrupt" fn keyboard_interrupt_handler(_stack_frame: InterruptStackFrame) {
    let mut port60: Port<u8> = Port::new(0x60);
    let scancode: u8 = unsafe { port60.read() };

    // Basic scancode to character mapping (very simplified, US keyboard layout)
    if let Some(c) = scancode_to_char(scancode) {
        COMMAND_LINE.lock().add_char(c); // Add character to command line
    }

    // Send End of Interrupt (EOI) to PIC
    unsafe {
        let mut pic1_command_port: Port<u8> = Port::new(0x20);
        pic1_command_port.write(0x20);
    }
}

fn scancode_to_char(scancode: u8) -> Option<char> {
    // Shift key state (for uppercase/symbols)
    static mut SHIFT_PRESSED: bool = false;

    match scancode {
        // Key press
        0x02 => Some(if unsafe { SHIFT_PRESSED } { '!' } else { '1' }),
        0x03 => Some(if unsafe { SHIFT_PRESSED } { '@' } else { '2' }),
        0x04 => Some(if unsafe { SHIFT_PRESSED } { '#' } else { '3' }),
        0x05 => Some(if unsafe { SHIFT_PRESSED } { '$' } else { '4' }),
        0x06 => Some(if unsafe { SHIFT_PRESSED } { '%' } else { '5' }),
        0x07 => Some(if unsafe { SHIFT_PRESSED } { '^' } else { '6' }),
        0x08 => Some(if unsafe { SHIFT_PRESSED } { '&' } else { '7' }),
        0x09 => Some(if unsafe { SHIFT_PRESSED } { '*' } else { '8' }),
        0x0A => Some(if unsafe { SHIFT_PRESSED } { '(' } else { '9' }),
        0x0B => Some(if unsafe { SHIFT_PRESSED } { ')' } else { '0' }),
        0x10 => Some(if unsafe { SHIFT_PRESSED } { 'Q' } else { 'q' }),
        0x11 => Some(if unsafe { SHIFT_PRESSED } { 'W' } else { 'w' }),
        0x12 => Some(if unsafe { SHIFT_PRESSED } { 'E' } else { 'e' }),
        0x13 => Some(if unsafe { SHIFT_PRESSED } { 'R' } else { 'r' }),
        0x14 => Some(if unsafe { SHIFT_PRESSED } { 'T' } else { 't' }),
        0x15 => Some(if unsafe { SHIFT_PRESSED } { 'Y' } else { 'y' }),
        0x16 => Some(if unsafe { SHIFT_PRESSED } { 'U' } else { 'u' }),
        0x17 => Some(if unsafe { SHIFT_PRESSED } { 'I' } else { 'i' }),
        0x18 => Some(if unsafe { SHIFT_PRESSED } { 'O' } else { 'o' }),
        0x19 => Some(if unsafe { SHIFT_PRESSED } { 'P' } else { 'p' }),
        0x1E => Some(if unsafe { SHIFT_PRESSED } { 'A' } else { 'a' }),
        0x1F => Some(if unsafe { SHIFT_PRESSED } { 'S' } else { 's' }),
        0x20 => Some(if unsafe { SHIFT_PRESSED } { 'D' } else { 'd' }),
        0x21 => Some(if unsafe { SHIFT_PRESSED } { 'F' } else { 'f' }),
        0x22 => Some(if unsafe { SHIFT_PRESSED } { 'G' } else { 'g' }),
        0x23 => Some(if unsafe { SHIFT_PRESSED } { 'H' } else { 'h' }),
        0x24 => Some(if unsafe { SHIFT_PRESSED } { 'J' } else { 'j' }),
        0x25 => Some(if unsafe { SHIFT_PRESSED } { 'K' } else { 'k' }),
        0x26 => Some(if unsafe { SHIFT_PRESSED } { 'L' } else { 'l' }),
        0x2C => Some(if unsafe { SHIFT_PRESSED } { 'Z' } else { 'z' }),
        0x2D => Some(if unsafe { SHIFT_PRESSED } { 'X' } else { 'x' }),
        0x2E => Some(if unsafe { SHIFT_PRESSED } { 'C' } else { 'c' }),
        0x2F => Some(if unsafe { SHIFT_PRESSED } { 'V' } else { 'v' }),
        0x30 => Some(if unsafe { SHIFT_PRESSED } { 'B' } else { 'b' }),
        0x31 => Some(if unsafe { SHIFT_PRESSED } { 'N' } else { 'n' }),
        0x32 => Some(if unsafe { SHIFT_PRESSED } { 'M' } else { 'm' }),
        0x39 => Some(' '),
        0x1C => Some('\n'),  // Enter key
        0x0E => Some('\x08'), // Backspace (ASCII 8)

        // Key release (for Shift, Ctrl, Alt)
        0x2A | 0x36 => {
            unsafe { SHIFT_PRESSED = true };
            None
        } // Left/Right Shift press
        0xAA | 0xB6 => {
            unsafe { SHIFT_PRESSED = false };
            None
        } // Left/Right Shift release

        _ => None,
    }
}

// --- PIC Initialization ---

fn init_pics() {
    // Initialize the Programmable Interrupt Controllers (PICs)

    let mut pic1_command: Port<u8> = Port::new(0x20);
    let mut pic1_data: Port<u8> = Port::new(0x21);
    let mut pic2_command: Port<u8> = Port::new(0xA0);
    let mut pic2_data: Port<u8> = Port::new(0xA1);

    // ICW1 (Initialization Command Word 1) - Begin initialization sequence
    unsafe {
        pic1_command.write(0x11); // ICW1: Edge triggered, Cascade mode, ICW4 needed
        pic2_command.write(0x11);
    }

    // ICW2 - Set interrupt vector offsets
    unsafe {
        pic1_data.write(0x20); // PIC1 vector offset (interrupts 0x20-0x27)
        pic2_data.write(0x28); // PIC2 vector offset (interrupts 0x28-0x2F)
    }

    // ICW3 - Configure cascading
    unsafe {
        pic1_data.write(4); // PIC1 connected to PIC2 via IRQ2 (bit 2 set)
        pic2_data.write(2); // PIC2 is cascaded (connected to IRQ2 on PIC1)
    }

    // ICW4 - Additional configuration
    unsafe {
        pic1_data.write(0x01); // 8086/88 (MCS-80/85) mode
        pic2_data.write(0x01);
    }

    // Mask all interrupts initially (except cascade on PIC1 and the slave PIC on PIC2)
    unsafe {
        pic1_data.write(0b1111_1100); // Unmask IRQ 0 (timer) and IRQ 1 (keyboard)
        pic2_data.write(0xFF);
    }
}

// --- Memory Management ---

// Simple Bump Allocator (for demonstration - replace with a real allocator)
pub struct BumpAllocator {
    heap_start: usize,
    heap_end: usize,
    next: AtomicU64,
}

impl BumpAllocator {
    /// Creates a new empty bump allocator.
    pub const fn new(heap_start: usize, heap_size: usize) -> Self {
        BumpAllocator {
            heap_start,
            heap_end: heap_start + heap_size,
            next: AtomicU64::new(heap_start as u64),
        }
    }
}

unsafe impl FrameAllocator<Size4KiB> for BumpAllocator {
    fn allocate_frame(&mut self) -> Option<PhysFrame> {
        let next = self.next.load(Ordering::Relaxed);
        if next + 4096 > self.heap_end as u64 {
            return None; // Out of memory
        }
        self.next.store(next + 4096, Ordering::Relaxed);
        Some(PhysFrame::containing_address(PhysAddr::new(next)))
    }
}

static mut HEAP_ALLOCATOR: BumpAllocator = BumpAllocator::new(0, 0);

#[alloc_error_handler]
fn alloc_error_handler(layout: alloc::alloc::Layout) -> ! {
    panic!("allocation error: {:?}", layout)
}

/// Initialize a new OffsetPageTable.
pub unsafe fn init_offset_page_table(physical_memory_offset: VirtAddr) -> OffsetPageTable<'static> {
    let level_4_table = active_level_4_table(physical_memory_offset);
    OffsetPageTable::new(level_4_table, physical_memory_offset)
}

unsafe fn active_level_4_table(physical_memory_offset: VirtAddr) -> &'static mut PageTable {
    let (level_4_table_frame, _) = Cr3::read();

    let phys = level_4_table_frame.start_address();
    let virt = physical_memory_offset + phys.as_u64();
    let page_table_ptr: *mut PageTable = virt.as_mut_ptr();

    &mut *page_table_ptr // unsafe
}

// --- Task Management ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TaskId(u64);

impl TaskId {
    fn new() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        TaskId(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}


struct Task {
    id: TaskId,
    stack: usize,       // Stack pointer (for simplicity)
    context: TaskContext, // Store context
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct TaskContext {
    rflags: u64,
    rdi: u64,
    rsi: u64,
    rbp: u64,
    rsp: u64,
    rip: u64,
}

impl TaskContext {
    fn new(entry_point: VirtAddr) -> Self {
        TaskContext {
            rflags: 0x202, // Set interrupt flag (bit 9) and reserved bit (bit 1)
            rdi: 0,
            rsi: 0,
            rbp: 0,    // Initial base pointer will be set to the top of the task's stack
            rsp: 0,    // Will be set later when the task is created
            rip: entry_point.as_u64(), // Set the entry point
        }
    }
}

impl Task {
    fn new(stack_start: usize, entry_point: VirtAddr) -> Self {
        let mut context = TaskContext::new(entry_point);
        context.rsp = stack_start as u64;   // Correct stack pointer
        context.rbp = stack_start as u64; // Stack Base Pointer
        Task {
            id: TaskId::new(),
            stack: stack_start,
            context,
        }
    }
}

lazy_static! {
    static ref TASKS: spin::Mutex<Vec<Task>> = spin::Mutex::new(Vec::new());
    static ref CURRENT_TASK_INDEX: spin::Mutex<usize> = spin::Mutex::new(0);
}

fn add_task(stack_start: usize, entry_point: VirtAddr) {
    let task = Task::new(stack_start, entry_point);
    println!("Task ID {} added, stack at: 0x{:x}, entry point: 0x{:x}", task.id.0, task.stack, task.context.rip);
    TASKS.lock().push(task);
}


fn switch_to_next_task(current_stack_frame: &mut InterruptStackFrame) {
    let mut tasks = TASKS.lock();
    let mut current_index = CURRENT_TASK_INDEX.lock();

    if tasks.is_empty() {
        return; // No tasks to switch to
    }

    // Save current task's context
    let current_task = &mut tasks[*current_index];
    current_task.context.rsp = current_stack_frame.stack_pointer.as_u64();
    current_task.context.rflags = current_stack_frame.flags.bits();
    current_task.context.rip = current_stack_frame.instruction_pointer.as_u64();
    current_task.context.rdi = current_stack_frame.general.rdi;
    current_task.context.rsi = current_stack_frame.general.rsi;
    current_task.context.rbp = current_stack_frame.general.rbp;

    // Get the next task
    *current_index = (*current_index + 1) % tasks.len();
    let next_task = &tasks[*current_index];

    // Restore next task's context
    current_stack_frame.stack_pointer = VirtAddr::new(next_task.context.rsp);
    current_stack_frame.flags =
        x86_64::registers::rflags::RFlags::from_bits_truncate(next_task.context.rflags);
    current_stack_frame.instruction_pointer = VirtAddr::new(next_task.context.rip);
    current_stack_frame.general.rdi = next_task.context.rdi;
    current_stack_frame.general.rsi = next_task.context.rsi;
    current_stack_frame.general.rbp = next_task.context.rbp;

    // println!("Switched to task {}", next_task.id.0); // Debug print
}

// A simple task (for demonstration)
fn task_1_entry() {
    let mut counter = 0;
    loop {
        println!("Task 1: {}", counter);
        counter += 1;
        if counter > 10 {
           break;
        }
          // Yield to the next task (using a system call)
        unsafe {
            asm!("int 0x80", in("rax") SYSCALL_YIELD);
        }
    }
      //Exit
    unsafe {
        asm!("int 0x80", in("rax") SYSCALL_EXIT);
    }
}

fn task_2_entry() {
    let mut counter = 0;
    loop {
        println!("Task 2: {}", counter);
        counter += 1;
        if counter > 10 {
           break;
        }
          // Yield to the next task (using a system call)
        unsafe {
            asm!("int 0x80", in("rax") SYSCALL_YIELD);
        }
    }
    //Exit
    unsafe {
        asm!("int 0x80", in("rax") SYSCALL_EXIT);
    }
}

extern "x86-interrupt" fn timer_interrupt_handler(stack_frame: &mut InterruptStackFrame) {
    // print!("."); // Show timer ticks (can be very frequent!)

    // Perform task switching (if needed)
    switch_to_next_task(stack_frame);

    // Send End of Interrupt (EOI) to the PIC
    unsafe {
        let mut pic1_command_port: Port<u8> = Port::new(0x20);
        pic1_command_port.write(0x20); // EOI command
    }
}

// --- Entry Point ---

fn update_cursor(x: u16, y: u16) {
    let pos = y * VGA_WIDTH as u16 + x;

    // Send the high byte of the cursor position
    unsafe {
        let mut command_port: Port<u8> = Port::new(0x3D4);
        let mut data_port: Port<u8> = Port::new(0x3D5);
        command_port.write(0x0E);  // Cursor Location High Byte
        data_port.write((pos >> 8) as u8);

        // Send the low byte of the cursor position
        command_port.write(0x0F); // Cursor Location Low Byte
        data_port.write(pos as u8);
    }
}

#[no_mangle]
pub extern "C" fn _start(boot_info: &'static BootInfo) -> ! {
    init_gdt();
    init_idt();
    init_pics();
    // Initialize the page table
    let physical_memory_offset = VirtAddr::new(boot_info.physical_memory_offset);
    let mut mapper = unsafe { init_offset_page_table(physical_memory_offset) };

    let mut frame_allocator = unsafe {
        BumpAllocator::new(boot_info.physical_memory_offset as usize, 0x100000) // Use bump allocator for now
    };
    // Initialize the heap.
    let heap_start = 0x_4444_4444_0000;
    let heap_size = 100 * 4096; // 100 pages

    let page_range = {
        let heap_start = VirtAddr::new(heap_start);
        let heap_end = heap_start + heap_size - 1u64; // Inclusive
        let heap_start_page = Page::containing_address(heap_start);
        let heap_end_page = Page::containing_address(heap_end);
        Page::range_inclusive(heap_start_page, heap_end_page)
    };

    for page in page_range {
        let frame = frame_allocator
            .allocate_frame()
            .expect("no more frames");
        let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;
        unsafe {
            mapper.map_to(page, frame, flags, &mut frame_allocator)
                .expect("map_to failed")
                .flush();
        }
    }

    // Initialize the global allocator (using the allocated heap)
    unsafe {
        HEAP_ALLOCATOR = BumpAllocator::new(heap_start as usize, heap_size as usize);
    }

    println!("Mini OS Booted!");

    // --- Task Creation ---
    // Allocate stacks for tasks
    let task_1_stack_start = 0x5000_0000;
    let task_1_stack_end = task_1_stack_start + 4096 * 5; // 5 pages for task 1 stack

    let task_2_stack_start = 0x6000_0000;
    let task_2_stack_end = task_2_stack_start + 4096 * 5; // 5 pages for task 2 stack

    // Map task stacks (similar to heap initialization)
    let task_1_stack_page_range = {
        let start = VirtAddr::new(task_1_stack_start);
        let end = VirtAddr::new(task_1_stack_end - 1);
        Page::range_inclusive(Page::containing_address(start), Page::containing_address(end))
    };

    let task_2_stack_page_range = {
        let start = VirtAddr::new(task_2_stack_start);
        let end = VirtAddr::new(task_2_stack_end - 1);
        Page::range_inclusive(Page::containing_address(start), Page::containing_address(end))
    };

    for page in task_1_stack_page_range {
        let frame = frame_allocator.allocate_frame().expect("no more frames");
        let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;
        unsafe {
            mapper.map_to(page, frame, flags, &mut frame_allocator)
                .expect("map_to failed")
                .flush();
        }
    }

    for page in task_2_stack_page_range {
        let frame = frame_allocator.allocate_frame().expect("no more frames");
        let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;
        unsafe {
            mapper.map_to(page, frame, flags, &mut frame_allocator)
                .expect("map_to failed")
                .flush();
        }
    }

    add_task(task_1_stack_end, VirtAddr::new(task_1_entry as u64));
    add_task(task_2_stack_end, VirtAddr::new(task_2_entry as u64));

    println!("Type 'help' for a list of commands.");
    print!("> ");

    unsafe {
        // Manually trigger a timer interrupt to start multitasking.
        // In a real scenario, the hardware timer would do this.
        asm!("int 0x20");
    }

    loop {
        x86_64::instructions::hlt();
    }
}