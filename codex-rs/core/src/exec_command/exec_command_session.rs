use std::sync::Arc;
use std::sync::Mutex as StdMutex;

use tokio::sync::broadcast;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

#[derive(Debug)]
pub(crate) struct ExecCommandSession {
    /// Queue for writing bytes to the process stdin (PTY master write side).
    writer_tx: mpsc::Sender<Vec<u8>>,
    /// Broadcast stream of output chunks read from the PTY. New subscribers
    /// receive only chunks emitted after they subscribe.
    output_tx: broadcast::Sender<Vec<u8>>,

    /// Child killer handle for termination on drop (can signal independently
    /// of a thread blocked in `.wait()`).
    killer: StdMutex<Option<Box<dyn portable_pty::ChildKiller + Send + Sync>>>,

    /// JoinHandle for the blocking PTY reader task.
    reader_handle: StdMutex<Option<JoinHandle<()>>>,

    /// JoinHandle for the stdin writer task.
    writer_handle: StdMutex<Option<JoinHandle<()>>>,

    /// JoinHandle for the child wait task.
    wait_handle: StdMutex<Option<JoinHandle<()>>>,

    /// Tracks whether the underlying process has exited.
    exit_status: std::sync::Arc<std::sync::atomic::AtomicBool>,

    /// Captures the exit code once the child terminates.
    exit_code: Arc<StdMutex<Option<i32>>>,
}

impl ExecCommandSession {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        writer_tx: mpsc::Sender<Vec<u8>>,
        output_tx: broadcast::Sender<Vec<u8>>,
        killer: Box<dyn portable_pty::ChildKiller + Send + Sync>,
        reader_handle: JoinHandle<()>,
        writer_handle: JoinHandle<()>,
        wait_handle: JoinHandle<()>,
        exit_status: std::sync::Arc<std::sync::atomic::AtomicBool>,
        exit_code: Arc<StdMutex<Option<i32>>>,
    ) -> (Self, broadcast::Receiver<Vec<u8>>) {
        let initial_output_rx = output_tx.subscribe();
        (
            Self {
                writer_tx,
                output_tx,
                killer: StdMutex::new(Some(killer)),
                reader_handle: StdMutex::new(Some(reader_handle)),
                writer_handle: StdMutex::new(Some(writer_handle)),
                wait_handle: StdMutex::new(Some(wait_handle)),
                exit_status,
                exit_code,
            },
            initial_output_rx,
        )
    }

    pub(crate) fn writer_sender(&self) -> mpsc::Sender<Vec<u8>> {
        self.writer_tx.clone()
    }

    pub(crate) fn output_receiver(&self) -> broadcast::Receiver<Vec<u8>> {
        self.output_tx.subscribe()
    }

    pub(crate) fn has_exited(&self) -> bool {
        self.exit_status.load(std::sync::atomic::Ordering::SeqCst)
    }

    pub(crate) fn exit_code(&self) -> Option<i32> {
        match self.exit_code.lock() {
            Ok(guard) => *guard,
            Err(_) => None,
        }
    }

    fn abort_or_detach_handle(handle: &StdMutex<Option<JoinHandle<()>>>, has_exited: bool) {
        let join_handle_opt = handle.lock().ok().and_then(|mut guard| guard.take());

        if let Some(join_handle) = join_handle_opt
            && !has_exited
        {
            join_handle.abort();
        }
        // When the process has already exited we intentionally drop the
        // handle without aborting so any in-flight writer flushes can
        // complete before the task finishes. Dropping the handle detaches
        // the task, allowing it to conclude naturally.
    }
}

impl Drop for ExecCommandSession {
    fn drop(&mut self) {
        let has_exited = self.has_exited();

        let killer_opt = self.killer.lock().ok().and_then(|mut stored| stored.take());

        if let Some(mut killer) = killer_opt
            && !has_exited
        {
            let _ = killer.kill();
        }

        Self::abort_or_detach_handle(&self.reader_handle, has_exited);
        Self::abort_or_detach_handle(&self.writer_handle, has_exited);
        Self::abort_or_detach_handle(&self.wait_handle, has_exited);
    }
}
